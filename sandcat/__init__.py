from __future__ import annotations

from pathlib import Path


__version__ = '0.1.0'

def get_examples_path() -> str:
    return str(Path(__file__).parent / 'examples')