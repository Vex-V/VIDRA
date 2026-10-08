"""Build and check the HTML docs. Needs nothing but vidra itself.

    python docs/pages/build.py           fill in every page's generated parts
    python docs/pages/build.py --check   fail when a page is stale, a link is
                                         broken, or a public name or parameter
                                         is missing from the page that covers it
    python docs/pages/build.py --run A.mp4 [B.mp4]
                                         also run every example on these videos,
                                         with free stand-ins for the hosted models
    ... --run A.mp4 B.mp4 --paid         with the models the examples name, and
                                         the keys in .env (costs money)

`docs/index.html` is the one page a reader opens; every other page, the
assets, the examples and this script live in `docs/pages/`. The pages are
written by hand. What this fills in, between `<!-- name -->` and
`<!-- /name -->` markers, is what would drift if it were copied: the head,
the top bar, the sidebar, the previous/next links, every example's code (from
pages/examples/), and the search index (pages/assets/search-index.js).
"""

from __future__ import annotations

import argparse
import html
import importlib
import inspect
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent
DOCS = PAGES_DIR.parent
ROOT = DOCS.parent
EXAMPLES = PAGES_DIR / "examples"
sys.path.insert(0, str(ROOT))

#: (path under docs/, title, group): the sidebar's order, and the
#: previous/next order.
PAGES = [
    ("index.html", "Start here", "Get started"),
    ("pages/concepts.html", "Concepts", "Get started"),
    ("pages/recipes.html", "Recipes", "Get started"),
    ("pages/pipeline.html", "The pipeline", "video_rag"),
    ("pages/live.html", "Live streams", "video_rag"),
    ("pages/stages.html", "Stages", "video_rag"),
    ("pages/samplers.html", "Samplers", "video_rag"),
    ("pages/questions.html", "Questions", "video_rag"),
    ("pages/search.html", "Search", "video_rag"),
    ("pages/aggregates.html", "Aggregates", "aggregates"),
    ("pages/definitions.html", "Custom prompts and profiles", "aggregates"),
    ("pages/models.html", "Models", "Bring your own"),
    ("pages/databases.html", "Databases", "Bring your own"),
    ("pages/reference.html", "Reference", "Reference"),
]

# ------------------------------------------------------------- generating

REGION = re.compile(r"<!-- (?P<name>[a-z]+)(?: (?P<arg>[\w.]+))? -->.*?<!-- /(?P=name) -->",
                    re.S)


def rel(current: str, target: str) -> str:
    """`target` (a path under docs/) as a link from the page `current`."""
    return posixpath.relpath(target, posixpath.dirname(current) or ".")


def head(current: str, title: str) -> str:
    root = rel(current, ".")
    root = "" if root == "." else root + "/"
    assets = rel(current, "pages/assets")
    return "\n".join([
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta name="docs-root" content="{root}">',
        f'<title>{html.escape(title)} · VIDRA docs</title>',
        f'<link rel="stylesheet" href="{assets}/style.css">',
        f'<script src="{assets}/search-index.js"></script>',
        f'<script src="{assets}/docs.js"></script>'])


def top(current: str) -> str:
    return "\n".join([
        f'<a class="brand" href="{rel(current, "index.html")}">VIDRA <span>docs</span></a>',
        '<button class="icon menu" id="menu" type="button" aria-label="Menu">&#9776;</button>',
        '<div class="spacer"></div>',
        '<div class="search"><input id="q" type="search" placeholder="Search the docs  ( / )" '
        'autocomplete="off" aria-label="Search the docs"><div class="results" id="results">'
        '</div></div>',
        '<button class="icon" id="theme" type="button" aria-label="Light or dark">&#9680;</button>'])


def nav(current: str) -> str:
    out, group = [], None
    for file, title, in_group in PAGES:
        if in_group != group:
            out.append(f'<div class="group">{html.escape(in_group)}</div>')
            group = in_group
        here = ' class="here"' if file == current else ""
        out.append(f'<a href="{rel(current, file)}"{here}>{html.escape(title)}</a>')
    return "\n".join(out)


def pager(current: str) -> str:
    files = [p[0] for p in PAGES]
    at = files.index(current)
    out = ['<div class="pager">']
    if at > 0:
        file, title, _ = PAGES[at - 1]
        out.append(f'<a href="{rel(current, file)}"><small>Previous</small>'
                   f'{html.escape(title)}</a>')
    if at < len(PAGES) - 1:
        file, title, _ = PAGES[at + 1]
        out.append(f'<a class="next" href="{rel(current, file)}"><small>Next</small>'
                   f'{html.escape(title)}</a>')
    out.append("</div>")
    return "\n".join(out)


def example(name: str) -> str:
    source = (EXAMPLES / name).read_text(encoding="utf-8").rstrip("\n")
    return (f'<div class="file">docs/pages/examples/{name}</div>\n'
            f'<pre><code class="python">{html.escape(source, quote=False)}</code></pre>')


def fill(file: str, text: str) -> str:
    title = next(t for f, t, _ in PAGES if f == file)

    def one(match: re.Match) -> str:
        name, arg = match["name"], match["arg"]
        body = {"head": lambda: head(file, title), "top": lambda: top(file),
                "nav": lambda: nav(file), "pager": lambda: pager(file),
                "example": lambda: example(arg)}.get(name)
        if body is None:
            return match[0]
        opening = f"<!-- {name} {arg} -->" if arg else f"<!-- {name} -->"
        return f"{opening}\n{body()}\n<!-- /{name} -->"
    return REGION.sub(one, text)


class Sections(HTMLParser):
    """The text of <main>, cut at every h2/h3 with an id, for the search index."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inside = 0
        self.sections: list[dict] = [{"id": "", "title": "", "text": []}]
        self.heading: list[str] | None = None
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main":
            self.inside += 1
        if not self.inside:
            return
        if tag in ("h1", "h2", "h3") and (attrs.get("id") or tag == "h1"):
            self.sections.append({"id": attrs.get("id", ""), "title": "", "text": []})
            self.heading = []
        if tag in ("script", "style") or "pager" in (attrs.get("class") or ""):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "main":
            self.inside -= 1
        if tag in ("h1", "h2", "h3") and self.heading is not None:
            self.sections[-1]["title"] = " ".join("".join(self.heading).split())
            self.heading = None
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.inside or self.skip:
            return
        if self.heading is not None:
            self.heading.append(data)
        else:
            self.sections[-1]["text"].append(data)


def search_index(pages: dict[str, str]) -> str:
    entries = []
    for file, title, _ in PAGES:
        parser = Sections()
        parser.feed(pages[file])
        for s in parser.sections:
            text = " ".join(" ".join(s["text"]).split())
            if not (s["title"] or text):
                continue
            entries.append({"page": title, "title": s["title"] or title,
                            "url": file + (f"#{s['id']}" if s["id"] else ""),
                            "text": text[:4000]})
    return ("// Generated by docs/pages/build.py: every section's text, for search.\n"
            "window.VIDRA_DOCS_INDEX = " + json.dumps(entries, ensure_ascii=False, indent=0)
            + ";\n")


def build() -> dict[Path, str]:
    """Every generated file's intended content."""
    pages = {}
    out: dict[Path, str] = {}
    for file, _, _ in PAGES:
        text = fill(file, (DOCS / file).read_text(encoding="utf-8"))
        pages[file] = text
        out[DOCS / file] = text
    out[PAGES_DIR / "assets" / "search-index.js"] = search_index(pages)
    return out


# --------------------------------------------------------------- checking

class Code(HTMLParser):
    """Every <code> element's text, and every id and href, of one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.code: list[str] = []
        self.ids: set[str] = set()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.hrefs.append(attrs["href"])
        if tag == "code":
            self.depth += 1
            self.code.append("")

    def handle_endtag(self, tag):
        if tag == "code" and self.depth:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth:
            self.code[-1] += data


#: Which page documents what: every parameter of each callable must appear in
#: code on that page. `Class.method` covers a method's parameters.
COVERAGE: dict[str, list[str]] = {
    "index.html": ["vidra.configure"],
    "pages/pipeline.html": [
        "vidra.video_rag.video_rag", "vidra.video_rag.process",
        "vidra.video_rag.Options", "vidra.video_rag.Run", "vidra.video_rag.validate",
        "vidra.video_rag.layout", "vidra.workflow.process", "vidra.workflow.Options",
        "vidra.workflow.validate", "vidra.workflow.extraction", "vidra.workflow.Run"],
    "pages/stages.html": [
        f"vidra.video_rag.{m}" for m in (
            "media.media", "media.split", "media.load",
            "audio.audio", "audio.listen", "audio.load",
            "boundaries.evidence", "boundaries.boundaries", "boundaries.detect",
            "boundaries.timeline", "boundaries.retune", "boundaries.calibrate",
            "boundaries.load",
            "video.video", "video.ingest", "video.load", "video.recreate",
            "video.FrameStore",
            "cut.cut", "cut.apply", "cut.load",
            "describe.describe", "describe.answer", "describe.load",
            "embed.embed", "embed.encode", "embed.load", "embed.Unit")],
    "pages/live.html": ["vidra.video_rag.video_rag_live", "vidra.video_rag.live.send",
                        "vidra.video_rag.search_observations"],
    "pages/samplers.html": [],
    "pages/questions.html": [f"vidra.video_rag.describe.{n}" for n in
                       ("add_question", "question", "questions", "remove_question")],
    "pages/search.html": ["vidra.video_rag.retrieve.search", "vidra.video_rag.retrieve.Moment"],
    "pages/aggregates.html": [f"vidra.aggregates.{n}" for n in (
        "record", "Record.excerpt", "Record.sightings", "Record.answer_ids",
        "stats", "coverage", "speakers", "ner", "sentiment", "summary", "chapters",
        "events", "entities", "custom", "aggregate", "validate", "answer", "load",
        "load_all", "load_input", "answers", "combine", "merge", "search",
        "available", "about", "tier_of", "kind_of", "settings_of", "uses_embedder",
        "takes_inputs", "build", "missing", "definition_rows")],
    "pages/definitions.html": [f"vidra.aggregates.{n}" for n in (
        "add_prompt", "add_profile", "remove_prompt", "remove_profile", "definition")],
    "pages/models.html": [f"vidra.{n}" for n in (
        "Models", "VLM.generate", "LLM.complete", "Embedder.embed",
        "Embedder.embed_query", "OpenAI", "Chat", "Anthropic", "Stub",
        "OpenAIEmbedder", "LocalEmbedder", "HashEmbedder")],
    "pages/databases.html": ["vidra.Supabase", "vidra.Folder"] + [
        f"vidra.Database.{n}" for n in (
            "write_media", "write_raw_transcript", "write_cuts", "write_timeline",
            "write_manifest", "write_transcript", "write_descriptions",
            "write_embedded", "write_prompts", "write_source", "write_answer",
            "write_aggregate_units", "write_definitions", "write_observations",
            "search", "search_aggregates", "search_observations", "spans",
            "video_ids", "close", "implements")],
    "pages/reference.html": [],
}

#: Modules whose every public name must appear in code on some page.
PUBLIC = ["vidra", "vidra.video_rag", "vidra.aggregates", "vidra.workflow"] + [
    f"vidra.video_rag.{m}" for m in
    ("media", "audio", "boundaries", "video", "cut", "describe", "embed", "retrieve")]

#: Parameters that are not settings a reader looks up.
UNLISTED = {"self", "cls"}


def resolve(dotted: str):
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(dotted)


def parameters(obj) -> list[str]:
    target = obj.__init__ if inspect.isclass(obj) and "__init__" in vars(obj) else obj
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return []
    return [p.name for p in signature.parameters.values()
            if p.name not in UNLISTED
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def check(built: dict[Path, str]) -> list[str]:
    problems = []
    for path, text in built.items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            problems.append(f"{path.relative_to(ROOT)} is stale: run python docs/pages/build.py")

    parsed: dict[str, Code] = {}
    for file, _, _ in PAGES:
        p = Code()
        p.feed(built[DOCS / file])
        parsed[file] = p

    def mentioned(file: str, word: str) -> bool:
        pattern = re.compile(rf"(?<![\w]){re.escape(word)}(?![\w])")
        return any(pattern.search(c) for c in parsed[file].code)

    # Every parameter, on the page that covers its callable.
    for file, names in COVERAGE.items():
        for dotted in names:
            obj = resolve(dotted)
            short = dotted.rsplit(".", 1)[-1]
            if not mentioned(file, short):
                problems.append(f"{file}: {dotted} is not mentioned in code")
            for name in parameters(obj):
                if not mentioned(file, name):
                    problems.append(f"{file}: {dotted} parameter `{name}` is not documented")

    # Every public name, somewhere.
    everywhere = [c for p in parsed.values() for c in p.code]
    for module in PUBLIC:
        for name in getattr(resolve(module), "__all__", []):
            if name.startswith("__"):
                continue
            if not any(re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", c) for c in everywhere):
                problems.append(f"{module}.{name} is public and appears on no page")

    # Every sampler, sampler setting, question and shape, on its page.
    from vidra.video_rag.core.sampling import samplers
    from vidra.video_rag.core.sampling.specs import SAMPLER_SETTINGS
    from vidra.video_rag.core.describe import prompts
    for name in [*samplers.available(), *SAMPLER_SETTINGS]:
        if not mentioned("pages/samplers.html", name):
            problems.append(f"pages/samplers.html: sampler or setting `{name}` is not documented")
    builtin = json.loads((ROOT / "vidra/video_rag/core/describe/prompts.json").read_text())
    for name in [*builtin["questions"], *builtin["shapes"]]:
        if not mentioned("pages/questions.html", name):
            problems.append(f"pages/questions.html: built-in `{name}` is not documented")
    del prompts

    # Links between pages land on a page and an id that exist.
    for file, p in parsed.items():
        for href in p.hrefs:
            if re.match(r"^[a-z]+:", href) or href.startswith("//"):
                continue
            page, _, anchor = href.partition("#")
            page = (posixpath.normpath(posixpath.join(posixpath.dirname(file), page))
                    if page else file)
            if page not in parsed:
                if not (DOCS / page).exists():
                    problems.append(f"{file}: link to missing {page}")
                continue
            if anchor and anchor not in parsed[page].ids:
                problems.append(f"{file}: link to {href}, which has no #{anchor}")

    for example_file in sorted(EXAMPLES.glob("*.py")):
        if not any(f"<!-- example {example_file.name} -->" in built[DOCS / f]
                   for f, _, _ in PAGES):
            problems.append(f"docs/pages/examples/{example_file.name} is shown on no page")
    return problems


# ---------------------------------------------------------------- running

#: Stand-ins for hosted models and Supabase, so every example runs for free.
FREE = r'''
import sys, runpy
import vidra.shared.models.llm as llm
import vidra.shared.models.embedders.remote as remote
import vidra.shared.storage.supabase as supabase
from vidra.shared.models.embedders import HashEmbedder
from vidra.shared.storage.folder import Folder

class Model(llm.Stub):
    def __init__(self, *args, **kwargs):
        super().__init__()

class Embedder(HashEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__()

class Database(Folder):
    def __init__(self, *args, **kwargs):
        super().__init__("data/out")

llm.OpenAI = llm.Anthropic = llm.Chat = Model
remote.OpenAIEmbedder = Embedder
supabase.Supabase = Database
'''

RUN = r'''
sys.argv = [sys.argv[1], *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
'''

#: In order: each example, its arguments, and whether it runs for free.
#: {folder} is the first video's output folder, which hosted.py writes.
RUNS = [
    ("hosted.py", ["{a}"], True),
    # These two read the folder hosted.py wrote (clip and yolo answers), so they
    # run before anything below re-runs that video with other samplers.
    ("aggregates_tour.py", ["{folder}"], True),
    ("custom_aggregate.py", ["{folder}"], True),
    ("stage_by_stage.py", ["{a}"], True),
    ("custom_question.py", ["{a}"], True),
    ("several_videos.py", ["{a}", "{b}"], True),
    ("custom_database.py", ["{a}"], True),
    ("custom_models.py", ["{a}"], False),
]


def run_examples(a: Path, b: Path, paid: bool) -> list[str]:
    problems = []
    work = Path(tempfile.mkdtemp(prefix="vidra-docs-"))
    env_file = ROOT / ".env"
    if paid and env_file.exists():
        shutil.copy(env_file, work / ".env")
    else:
        (work / ".env").write_text("")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "VIDRA_DATA": str(work / "data"),
           "PYTHONIOENCODING": "utf-8"}
    values = {"a": str(a.resolve()), "b": str(b.resolve()),
              "folder": str(work / "data" / "out" / a.stem)}
    try:
        for name, args, free in RUNS:
            if not (free or paid):
                print(f"skip  {name} (paid only)")
                continue
            argv = [str(EXAMPLES / name), *[arg.format(**values) for arg in args]]
            prelude = ("" if paid else FREE) + "import sys, runpy\n" + RUN
            done = subprocess.run([sys.executable, "-c", prelude, *argv], cwd=work, env=env,
                                  capture_output=True, text=True, encoding="utf-8")
            if done.returncode == 0:
                print(f"ok    {name}")
            else:
                tail = "\n".join(done.stderr.strip().splitlines()[-12:])
                print(f"FAIL  {name}\n{tail}\n")
                problems.append(f"example {name} failed")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", nargs="+", type=Path, metavar="VIDEO")
    parser.add_argument("--paid", action="store_true")
    options = parser.parse_args()

    built = build()
    if options.check:
        problems = check(built)
    else:
        for path, text in built.items():
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8", newline="\n")
                print("wrote", path.relative_to(ROOT))
        problems = []
    if options.run:
        a = options.run[0]
        b = options.run[1] if len(options.run) > 1 else options.run[0]
        problems += run_examples(a, b, options.paid)
    for problem in problems:
        print("problem:", problem)
    if options.check or options.run:
        print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
