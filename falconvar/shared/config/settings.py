"""`falconvar.configure()`: what a process sets once, before anything runs.

    import falconvar
    falconvar.configure(data_root="/var/lib/falconvar",   # where runs write
                        weights="/models",                # checkpoints we fetch
                        env_file=".env",                  # read keys from here
                        hf_token=vault.get("hf"))         # gated downloads

The library never reads `.env` by itself; `env_file` asks for one. The
Hugging Face token is for downloading gated weights (pyannote) and is held in
this process only; without one, Hugging Face's own lookup applies (`HF_TOKEN`,
then `hf auth login`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import env, paths
from ..reporting.errors import Refused

_hf_token: Optional[str] = None


def configure(data_root: Optional[Path | str] = None,
              weights: Optional[Path | str] = None,
              env_file: Optional[Path | str] = None,
              hf_token: Optional[str] = None) -> None:
    """Say where this process writes, which `.env` it reads and how it downloads.
    Each argument left None changes nothing. `data_root` and `weights` outrank
    `FALCONVAR_DATA` / `FALCONVAR_WEIGHTS`; `env_file` is read at once and never
    overrides a variable already set.
    """
    global _hf_token
    paths.configure(data_root=data_root, weights=weights)
    if env_file is not None:
        if not Path(env_file).expanduser().is_file():
            # A named file that does not exist is refused.
            raise Refused(f"env_file {env_file} does not exist")
        env.load(env_file, force=True)
    if hf_token is not None:
        if not str(hf_token).strip():
            raise Refused("hf_token must not be empty; leave it out to use "
                          "HF_TOKEN or `hf auth login`")
        _hf_token = str(hf_token).strip()


def hf_token() -> Optional[str]:
    """The token `configure` was given, or None for Hugging Face's own lookup."""
    return _hf_token


__all__ = ["configure", "hf_token"]
