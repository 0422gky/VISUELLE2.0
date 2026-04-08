"""
Thin launcher: runs PJ_nan/similarity_wape_pipeline.py with the same CLI (including
--train_emb_npy / --test_emb_npy). Run from anywhere:

  python run_similarity_wape_with_projected.py --train_csv ... --test_csv ... \\
      --train_emb_npy results/curve_projector/projected/train_item_embeddings_projected.npy \\
      --test_emb_npy results/curve_projector/projected/test_item_embeddings_projected.npy
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    # GTM-Transformer -> GTM -> PJ_nan
    pj_nan = Path(__file__).resolve().parents[2]
    script = pj_nan / "similarity_wape_pipeline.py"
    if not script.is_file():
        raise FileNotFoundError(f"Expected {script}")
    runpy.run_path(str(script), run_name="__main__")
