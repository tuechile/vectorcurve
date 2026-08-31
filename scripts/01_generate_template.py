#!/usr/bin/env python
"""
Generate the printable capture sheet.

Usage:
    python scripts/01_generate_template.py --charset abcdefghijklmnopqrstuvwxyz \
        --cols 6 --cell-mm 30 --out template.svg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.template import generate_template, DEFAULT_CHARSET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--cell-mm", type=float, default=30)
    parser.add_argument("--margin-mm", type=float, default=15)
    parser.add_argument("--out", default="template.svg")
    args = parser.parse_args()

    meta = generate_template(
        charset=args.charset,
        cols=args.cols,
        cell_mm=args.cell_mm,
        margin_mm=args.margin_mm,
        out_path=args.out,
    )

    meta_path = Path(args.out).with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.out} ({meta['rows']} rows x {meta['cols']} cols, "
          f"{meta['cell_mm']}mm cells). Grid metadata saved to {meta_path}.")
    print("Print this at 100% scale (no 'fit to page'), write one glyph per box, then scan.")


if __name__ == "__main__":
    main()
