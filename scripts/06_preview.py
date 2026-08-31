#!/usr/bin/env python
"""
Render every trained glyph model in a directory as a contact sheet, so you
can eyeball curve quality (or spot degenerate/collapsed glyphs) before
spending a font build on them.

Usage:
    python scripts/06_preview.py --models-dir models --out outputs/glyph_preview.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.curve_model import eval_dense, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="outputs/glyph_preview.png")
    parser.add_argument("--cols", type=int, default=6)
    args = parser.parse_args()

    model_paths = sorted(Path(args.models_dir).glob("*.pt"))
    if not model_paths:
        print(f"No .pt files found in {args.models_dir}/. Run 03_train_glyphs.py first.")
        sys.exit(1)

    cols = min(args.cols, len(model_paths))
    rows = -(-len(model_paths) // cols)  # ceil div
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = [axes] if rows == 1 and cols == 1 else np.asarray(axes).flatten()

    for ax, path in zip(axes, model_paths):
        model = load_model(path)
        _, pts = eval_dense(model, num_points=300)
        ax.plot(pts[:, 0], pts[:, 1], "k-", linewidth=1.8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(path.stem, fontsize=10)
        ax.axis("off")

    for ax in axes[len(model_paths):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Wrote {out_path} ({len(model_paths)} glyphs)")


if __name__ == "__main__":
    main()
