"""Two data roots in one process.

`falconvar.configure()` says where *this process* reads and writes, which is
the right answer for an application that owns its own deployment and the
wrong one for anything serving more than one. A `Workspace` is the same
statement scoped to a call instead:

    ws = Workspace(data_root="/var/lib/tenant-a")
    ws.media.run("talk.mp4")                 # bound to this workspace

    with Workspace(data_root="/var/lib/tenant-b"):
        media.run("talk.mp4")                # module-level call, other root

Both spellings do the same thing -- the second is for code that already calls
the components by module, and the first for code that would rather not think
about a context being active.

**It is a `ContextVar`, not a global.** A `ContextVar` is already per-thread
and per-asyncio-task, so a workspace entered on one task is invisible to
another; a plain module global would be a data race the moment `describe`
gathers its calls. Nesting is `reset(token)`, which is exact, rather than
"restore whatever I saw on the way in", which is not.

**Precedence.** The active workspace, then `configure()`, then
`FALCONVAR_DATA` / `FALCONVAR_WEIGHTS`, then the checkout, then
`~/.falconvar`. A workspace beats `configure()` because it is the narrower
statement, made by the code making the call.

**One honest limitation: API keys are process-global.** `env_file=` reads that
file, but keys land in `os.environ`, and one process has one of those. Two
workspaces needing *different* values for `OPENAI_API_KEY` is not supported,
and saying so beats a parameter that looks like it works. What a workspace
does isolate is everything under the data root, which includes the two
hand-editable vocabularies -- `prompts.json` and `aggregates.json` -- whose
caches are keyed by that root for exactly this reason.

**Nothing here is needed for the pure verbs.** `split`, `listen`, `detect`,
`timeline`, `ingest`, `apply`, `answer` and `encode` take their inputs and
return their outputs; they resolve no root at all, so there is nothing for a
workspace to scope. This is for the `run`/`load` half.
"""

from __future__ import annotations

from contextvars import Token
from pathlib import Path
from typing import Any, Optional

from .shared import paths

#: The components a workspace binds, and the module each one lives in. The
#: same list the API dispatches over: adding a component adds a row, not a
#: property.
COMPONENTS = ("media", "audio", "boundaries", "video", "cut", "describe",
              "embed", "retrieve")


class _Bound:
    """One component, with this workspace entered around every call.

    A proxy rather than eight wrapped functions, because a component's surface
    is whatever it exports -- `run`, `load`, the pure verb, the registries,
    the errors -- and restating that here is the drift `/capabilities` exists
    to avoid. An attribute that is not callable comes back untouched, so
    `ws.boundaries.POLICIES` is the table and not a wrapper around it.
    """

    __slots__ = ("_workspace", "_name", "_module")

    def __init__(self, workspace: "Workspace", name: str) -> None:
        self._workspace = workspace
        self._name = name
        self._module: Any = None

    def _resolve(self) -> Any:
        if self._module is None:
            # Imported on first use, not when the workspace is made: binding
            # eight components would import 53 modules and `av` for someone
            # who wanted one of them.
            from importlib import import_module
            self._module = import_module(f".video_rag.{self._name}", __package__)
        return self._module

    def __getattr__(self, attribute: str) -> Any:
        value = getattr(self._resolve(), attribute)
        if not callable(value) or isinstance(value, type):
            return value

        def inside(*args: Any, **kwargs: Any) -> Any:
            with self._workspace:
                return value(*args, **kwargs)

        inside.__name__ = getattr(value, "__name__", attribute)
        inside.__doc__ = getattr(value, "__doc__", None)
        return inside

    def __dir__(self) -> list[str]:
        return dir(self._resolve())

    def __repr__(self) -> str:
        return f"<{self._name} in {self._workspace}>"


class Workspace:
    """Where one body of work reads and writes.

    `data_root` and `weights` default to whatever the process would resolve
    anyway, so `Workspace()` is the ambient configuration made explicit rather
    than a change to it.
    """

    def __init__(self, data_root: Optional[Path | str] = None,
                 weights: Optional[Path | str] = None,
                 env_file: Optional[Path | str] = None) -> None:
        self.roots: dict[str, Path] = {}
        for key, value in (("data", data_root), ("weights", weights)):
            if value is not None:
                self.roots[key] = Path(value).expanduser().resolve()
        self.env_file = Path(env_file).expanduser() if env_file else None
        self._tokens: list[Token] = []

        for name in COMPONENTS:
            setattr(self, name, _Bound(self, name))

    # ------------------------------------------------------------ the roots

    def data_root(self) -> Path:
        with self:
            return paths.data_root()

    def out_root(self) -> Path:
        with self:
            return paths.out_root()

    def videos(self) -> list[str]:
        """Every video under this workspace's output root."""
        with self:
            return paths.videos()

    # ----------------------------------------------------------- the context

    def __enter__(self) -> "Workspace":
        self._tokens.append(paths._active.set(self.roots))
        if self.env_file is not None:
            # Imported here, not at module scope: `shared.env` pulls in
            # `python-dotenv`, measured at ~20 ms, and `import falconvar`
            # binds this module. A workspace with no `env_file` never pays it.
            # `force`, because `env.load` remembers having read one already --
            # and this is a deliberate request for a different file.
            from .shared import env
            env.load(self.env_file, force=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        # `reset` on the token this `__enter__` produced, so nesting unwinds
        # exactly. A list because the same workspace may legitimately be
        # entered twice -- `ws.media.run` enters it, and a caller may already
        # be inside a `with`.
        paths._active.reset(self._tokens.pop())

    def __repr__(self) -> str:
        where = self.roots.get("data", "<ambient>")
        return f"Workspace(data_root={where})"


__all__ = ["COMPONENTS", "Workspace"]
