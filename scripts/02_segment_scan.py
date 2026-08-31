#!/usr/bin/env python
"""
Crop a scanned capture sheet into one PNG per glyph, using the grid
geometry recorded by 01_generate_template.py.

Usage:
    python scripts/02_segment_scan.py --scan scan.png --template-meta template.json \
        --dpi 300 --out-dir data/glyphs

Assumes the scan is aligned to the printed page (flatbed scanner, or a
very carefully squared-up photo) - no perspective correction is done.

--inset-mm shrinks each crop inward from the cell boundary before saving,
so the printed cell border (template.py's stroke-width="0.2" rect) isn't
included in the crop. Without this, threshold_otsu/skeletonize downstream
(pipeline/preprocess.py) treat that border as ink alongside the actual
letter - and since it's a second, disconnected skeleton component whose
corner-rounding leaves spur artifacts, it can hijack the arc-length walk's
start vertex and produce a garbage traced curve instead of the letter.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from skimage.io import imread, imsave

MM_PER_INCH = 25.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)
    parser.add_argument("--template-meta", required=True)
    parser.add_argument("--dpi", type=float, default=300)
    parser.add_argument("--out-dir", default="data/glyphs")
    parser.add_argument(
        "--inset-mm", type=float, default=2.0,
        help="shrink each crop inward by this many mm to exclude the printed cell border",
    )
    args = parser.parse_args()

    with open(args.template_meta) as f:
        meta = json.load(f)

    img = imread(args.scan, as_gray=True)
    px_per_mm = args.dpi / MM_PER_INCH
    cell_px = meta["cell_mm"] * px_per_mm
    margin_px = meta["margin_mm"] * px_per_mm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = img.shape
    expected_w = 2 * margin_px + meta["cols"] * cell_px
    expected_h = 2 * margin_px + meta["rows"] * cell_px
    if abs(w - expected_w) / expected_w > 0.05 or abs(h - expected_h) / expected_h > 0.05:
        print(f"WARNING: scan size {w}x{h}px doesn't closely match expected "
              f"{expected_w:.0f}x{expected_h:.0f}px at {args.dpi} DPI. "
              f"Check the scan DPI matches --dpi, and that it was printed at 100% scale.")

    inset_px = args.inset_mm * px_per_mm
    if inset_px * 2 >= cell_px:
        print(f"ERROR: --inset-mm {args.inset_mm} is too large for cell_mm {meta['cell_mm']}.")
        sys.exit(1)

    written = []
    for i, char in enumerate(meta["charset"]):
        row, col = divmod(i, meta["cols"])
        x0 = int(margin_px + col * cell_px + inset_px)
        y0 = int(margin_px + row * cell_px + inset_px)
        x1 = int(x0 + cell_px - 2 * inset_px)
        y1 = int(y0 + cell_px - 2 * inset_px)
        crop = img[y0:y1, x0:x1]

        safe_name = char if char.isalnum() else f"u{ord(char):04x}"
        out_path = out_dir / f"{safe_name}.png"
        imsave(out_path, (crop * 255).astype(np.uint8) if crop.dtype != np.uint8 else crop)
        written.append(str(out_path))

    print(f"Wrote {len(written)} glyph crops to {out_dir}/")


if __name__ == "__main__":
    main()
