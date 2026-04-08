"""
Apply trained CurveMetricProjector (+ optional PCA) to train/test embedding .npy files.
Writes projected L2-normalized vectors as *_item_embeddings_projected.npy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.curve_metric_projector import CurveMetricProjector, transform_with_pca


@torch.no_grad()
def project_matrix(
    emb: np.ndarray,
    projector_dir: Path,
    device: torch.device,
) -> np.ndarray:
    config_path = projector_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    pca_path = projector_dir / "pca_model.joblib"
    if cfg.get("use_pca"):
        pca = joblib.load(pca_path)
        X = transform_with_pca(pca, emb)
    else:
        X = emb.astype(np.float32)

    try:
        ckpt = torch.load(projector_dir / "projector.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(projector_dir / "projector.pt", map_location=device)
    input_dim = ckpt.get("input_dim", X.shape[1])
    hidden_dim = cfg.get("hidden_dim", 128)
    out_dim = cfg.get("output_dim", 64)

    model = CurveMetricProjector(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    x_t = torch.from_numpy(X).float().to(device)
    z = model(x_t).cpu().numpy().astype(np.float32)
    return z


def main():
    parser = argparse.ArgumentParser(description="Apply curve metric projector to embeddings")
    parser.add_argument("--projector_dir", type=str, required=True, help="Directory with projector.pt, config.json, optional pca_model.joblib")
    parser.add_argument("--train_embeddings_npy", type=str, default="", help="Input train (N, D) frozen GTM")
    parser.add_argument("--test_embeddings_npy", type=str, default="", help="Input test (M, D)")
    parser.add_argument("--output_dir", type=str, default="", help="Default: projector_dir/projected")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    projector_dir = Path(args.projector_dir)
    out_dir = Path(args.output_dir) if args.output_dir else (projector_dir / "projected")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.train_embeddings_npy:
        tr = np.load(args.train_embeddings_npy)
        z_tr = project_matrix(tr, projector_dir, device)
        np.save(out_dir / "train_item_embeddings_projected.npy", z_tr)
        print(f"Wrote {out_dir / 'train_item_embeddings_projected.npy'} shape={z_tr.shape}")

    if args.test_embeddings_npy:
        te = np.load(args.test_embeddings_npy)
        z_te = project_matrix(te, projector_dir, device)
        np.save(out_dir / "test_item_embeddings_projected.npy", z_te)
        print(f"Wrote {out_dir / 'test_item_embeddings_projected.npy'} shape={z_te.shape}")

    if not args.train_embeddings_npy and not args.test_embeddings_npy:
        print("Provide at least one of --train_embeddings_npy or --test_embeddings_npy")


if __name__ == "__main__":
    main()
