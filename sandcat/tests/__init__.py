from __future__ import annotations

from pathlib import Path


def get_plumed_path() -> str:
    return str(Path(__file__).parent / 'plumed2')