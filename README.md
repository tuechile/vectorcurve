# vectorcurve

Turn a page of your own handwriting into an installable font — plus, as a
byproduct, a tool for morphing one person's handwriting into another's.

This builds directly on findings from an earlier research project,
[AM111_Final](https://github.com/tuechile/AM111_Final): a small neural
network trained per-glyph on a curve's arc-length parameterization can
represent handwriting strokes — including self-intersecting loops — more
compactly and robustly than classical splines, *if* it mixes a `sin`
activation (for high-frequency, loop-capable detail) with `tanh` (to keep
the fit smooth rather than noisy). That project also introduced a
K-control-point mechanism: instead of fitting every raw pixel, a small
learned basis picks out the K points that matter most for the curve's
shape.

`vectorcurve` reuses that exact mechanism, but points it at a real
deliverable: a compact, trained representation of each letter you write,
exported as actual font glyph outlines.

## Why a neural curve fit instead of a normal vectorizer

Generic raster-to-vector tools (potrace, Adobe Image Trace, Vector Magic)
already turn scanned handwriting into vector curves. They're not the
point here. What they don't do well is handle **cursive self-intersections**
gracefully — a plain polyline trace or a spline fit to a noisy skeleton
tends to draw a straight "shortcut" segment across a loop instead of
following it. The sin/tanh neural curve exists specifically to avoid that
artifact (see the AM111 writeup, §4.1 and §7.1), which matters a lot for
letterforms like cursive `e`, `o`, `l`.

## Pipeline

```
scan/photo of handwriting
        │  (skimage: Otsu threshold -> skeletonize -> Eulerian trail -> arc-length param)
        ▼
   ordered (s, x, y) curve per glyph
        │  (pipeline/curve_model.py: K-control-point sin+tanh neural basis, trained per glyph)
        ▼
   compact NeuralBasisCurve per glyph
        │
        ├─(pipeline/font_export.py)──▶ closed glyph outline (Shapely stroke buffer) ──▶ .ttf
        │
        └─(pipeline/morph.py, byproduct)──▶ weight-interpolated in-between curves ──▶ .gif
```

## Usage

```bash
pip install -r requirements.txt

# 1. Generate a printable capture sheet (one box per letter)
python scripts/01_generate_template.py --out template.svg

# 2. Print at 100% scale, write one letter per box, scan at (e.g.) 300 DPI
python scripts/02_segment_scan.py --scan scan.png --template-meta template.json \
    --dpi 300 --out-dir data/glyphs

# 3. Train a compact neural curve for each glyph
python scripts/03_train_glyphs.py --glyphs-dir data/glyphs --out-dir models --k 60 --epochs 3000

# 4. Export an installable font
python scripts/04_export_font.py --models-dir models --out fonts/MyHandwriting.ttf \
    --family-name "My Handwriting"

# 5. (Byproduct) morph two trained samples of the same letter into each other
python scripts/05_make_morph.py --model-a models/a.pt --model-b models/a_friend.pt \
    --out outputs/morph_a.gif
```

## v1 limitations / next steps

- **Scan alignment**: cropping assumes a flatbed-scanner-quality scan aligned
  to the printed template. A phone photo needs perspective correction first
  (not implemented).
- **Polygonal outlines**: glyph contours come from a Shapely stroke buffer
  and are many short line segments, not fitted cubic/quadratic Beziers.
  They render fine but aren't as compact as a proper Bezier fit would be —
  a natural follow-up.
- **No live UI yet**: this is a local script pipeline by design for v1. A
  small web front end (upload scan → preview glyphs → download font) is a
  reasonable v2.

## Credit

Neural curve fitting approach and the K-point compression idea are from
[`AM111_Final`](https://github.com/tuechile/AM111_Final), a final project
for Harvard APMTH 111 (Introduction to Scientific Computing).
