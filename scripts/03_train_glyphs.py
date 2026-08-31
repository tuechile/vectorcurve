#!/usr/bin/env python
"""
Train a NeuralBasisCurve for every glyph image in a directory.

Usage:
    python scripts/03_train_glyphs.py --glyphs-dir data/glyphs --out-dir models \
        --k 60 --epochs 3000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skimage.io import imread
from skimage.util import img_as_float

from pipeline.preprocess import image_to_curve
from pipeline.curve_model import train_neural_basis, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glyphs-dir", default="data/glyphs")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--k", type=int, default=60, help="control points per glyph")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    glyphs_dir = Path(args.glyphs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glyphs_dir.glob("*.png"))
    if not image_paths:
        print(f"No .png files found in {glyphs_dir}/. Run 02_segment_scan.py first.")
        sys.exit(1)

    results = []
    for path in image_paths:
        char = path.stem
        print(f"Training '{char}' from {path.name} ...")
        img = img_as_float(imread(path, as_gray=True))
        try:
            s, coords, _ = image_to_curve(img)
        except ValueError as e:
            print(f"  skipped: {e}")
            continue

        model, mse = train_neural_basis(
            s, coords, k=args.k, num_epochs=args.epochs, lr=args.lr, verbose=False,
        )
        out_path = out_dir / f"{char}.pt"
        save_model(model, out_path)
        results.append((char, mse))
        print(f"  saved {out_path} (final MSE={mse:.6e})")

    print("\nDone.")
    for char, mse in results:
        print(f"  {char}: MSE={mse:.6e}")


if __name__ == "__main__":
    main()
