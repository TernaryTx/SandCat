from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LigFile:
    path: str
    use_in_pipeline: bool = True

    def __post_init__(self):
        """Validate that path ends with allowed extensions."""
        allowed_extensions = {'.json', '.sdf', '.sdf.gz'}
        assertion_error_text = f"LigFile path must end with one of: {', '.join(allowed_extensions)}"
        assert any([
            str(self.path).endswith(ext)
            for ext in allowed_extensions
        ]), assertion_error_text
