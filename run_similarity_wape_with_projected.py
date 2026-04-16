"""
Thin launcher for similarity_wape_pipeline.py.
It forwards the exact same CLI (including --train_emb_npy / --test_emb_npy).
"""

from __future__ import annotations

import runpy
from pathlib import Path


def _resolve_pipeline_script() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "similarity_wape_pipeline.py",
        here.parent / "similarity_wape_pipeline.py",
    ]
    for script in candidates:
        if script.is_file():
            return script
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot locate similarity_wape_pipeline.py. Tried:\n{tried}")


if __name__ == "__main__":
    script = _resolve_pipeline_script()
    runpy.run_path(str(script), run_name="__main__")
