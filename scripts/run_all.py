#!/usr/bin/env python
"""
Run the pipeline end-to-end: segment scan -> train glyphs -> export font ->
preview. Each stage is just this project's own numbered script, invoked in
order and stopping at the first failure.

Doesn't include 01_generate_template.py (a one-time setup step - print it,
write on it, scan it, before this can run at all) or 05_make_morph.py (the
byproduct, needs two specific glyph models chosen by hand, not a natural
"run everything" step).

Usage (matches the README's individual-script defaults):
    python scripts/run_all.py --scan scan.png --template-meta template.json \
        --dpi 300 --family-name "My Handwriting"

Already have data/glyphs/ cropped and just want to retrain/export/preview:
    python scripts/run_all.py --skip-segment --family-name "My Handwriting"
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_stage(title, args):
    print(f"\n=== {title} ===")
    print(f"$ {' '.join(str(a) for a in args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", default="scan.png")
    parser.add_argument("--template-meta", default="template.json")
    parser.add_argument("--dpi", type=float, default=300)
    parser.add_argument("--inset-mm", type=float, default=2.0)
    parser.add_argument("--glyphs-dir", default="data/glyphs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--k", type=int, default=60, help="control points per glyph")
    parser.add_argument("--degree", type=int, default=3, help="B-spline degree (default cubic)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--font-out", default="fonts/MyHandwriting.ttf")
    parser.add_argument("--family-name", default="My Handwriting")
    parser.add_argument("--preview-out", default="outputs/glyph_preview.png")
    parser.add_argument(
        "--skip-segment", action="store_true",
        help="skip re-cropping the scan; use whatever's already in --glyphs-dir",
    )
    args = parser.parse_args()

    if not args.skip_segment:
        run_stage("1/3 Segmenting scan into glyph crops", [
            SCRIPTS_DIR / "02_segment_scan.py",
            "--scan", args.scan,
            "--template-meta", args.template_meta,
            "--dpi", str(args.dpi),
            "--inset-mm", str(args.inset_mm),
            "--out-dir", args.glyphs_dir,
        ])
    else:
        print("\n=== 1/3 Segmenting scan into glyph crops (skipped) ===")

    run_stage("2/3 Training a B-spline curve per glyph", [
        SCRIPTS_DIR / "03_train_glyphs.py",
        "--glyphs-dir", args.glyphs_dir,
        "--out-dir", args.models_dir,
        "--k", str(args.k),
        "--degree", str(args.degree),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ])

    run_stage("3/3 Exporting font", [
        SCRIPTS_DIR / "04_export_font.py",
        "--models-dir", args.models_dir,
        "--out", args.font_out,
        "--family-name", args.family_name,
    ])

    print("\n=== Rendering glyph preview (sanity check) ===")
    run_stage("Preview", [
        SCRIPTS_DIR / "06_preview.py",
        "--models-dir", args.models_dir,
        "--out", args.preview_out,
    ])

    print(f"\nDone. Font: {args.font_out}  Preview: {args.preview_out}")


if __name__ == "__main__":
    main()
