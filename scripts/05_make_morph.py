#!/usr/bin/env python
"""
Byproduct, not the main deliverable: morph one trained glyph curve into
another (same letter, two different writers or two points in time) and
render it as a looping GIF.

Usage:
    python scripts/05_make_morph.py --model-a models/a.pt --model-b models/a_friend.pt \
        --out outputs/morph_a.gif

Both models must have been trained with the same --k.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.curve_model import load_model
from pipeline.morph import render_morph_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--out", default="outputs/morph.gif")
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_morph_gif(model_a, model_b, out_path, n_frames=args.frames)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
