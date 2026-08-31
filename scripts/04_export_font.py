#!/usr/bin/env python
"""
Build an installable TTF from trained per-glyph models.

Usage:
    python scripts/04_export_font.py --models-dir models --out fonts/MyHandwriting.ttf \
        --family-name "My Handwriting" --stroke-width 0.06
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.curve_model import load_model
from pipeline.font_export import build_font


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default="fonts/MyHandwriting.ttf")
    parser.add_argument("--family-name", default="My Handwriting")
    parser.add_argument("--stroke-width", type=float, default=0.06,
                         help="stroke thickness as a fraction of the glyph's own [0,1] extent")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    char_to_model = {}
    for path in sorted(models_dir.glob("*.pt")):
        char = path.stem
        if len(char) != 1:
            continue  # skip non-single-char files (e.g. morph variants like "a_friend.pt")
        char_to_model[char] = load_model(path)

    if not char_to_model:
        print(f"No single-character .pt models found in {models_dir}/. Run 03_train_glyphs.py first.")
        sys.exit(1)

    out_path = build_font(
        char_to_model,
        output_path=args.out,
        family_name=args.family_name,
        stroke_width_frac=args.stroke_width,
    )
    print(f"Wrote {out_path} with {len(char_to_model)} glyphs: {sorted(char_to_model)}")


if __name__ == "__main__":
    main()
