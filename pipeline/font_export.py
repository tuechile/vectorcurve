"""
Trained NeuralBasisCurve glyphs -> an installable TTF font.

A font glyph needs a *closed, fillable* outline, not an open stroke
centerline. We turn each glyph's dense curve into a ribbon-shaped polygon
by buffering the polyline with Shapely (round caps/joins), which also
naturally produces the interior "counter" hole for closed loops like
'o' or 'e' - no separate hole-detection logic needed. The polygon's
exterior/interior rings become the TrueType glyf contours directly.

This keeps outlines polygonal (many short line segments) rather than
fitting cubic/quadratic Beziers to them. That's a deliberate v1
simplification - see README for the Bezier-fit follow-up.
"""

from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry.polygon import orient
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from .curve_model import eval_dense

UNITS_PER_EM = 1000
ASCENT = 800
DESCENT = -200
SIDE_BEARING_FRAC = 0.08  # fraction of em added as left/right margin


def _polygon_to_contours(polygon):
    """TrueType convention: outer contours clockwise, holes counter-clockwise."""
    polygon = orient(polygon, sign=-1.0)  # exterior -> clockwise
    contours = [list(polygon.exterior.coords)[:-1]]
    for interior in polygon.interiors:
        contours.append(list(interior.coords)[:-1])
    return contours


def curve_to_glyph(points_xy, stroke_width_frac=0.06, units_per_em=UNITS_PER_EM):
    """
    points_xy: (N,2) dense curve in normalized [0,1]^2 coords (y up).
    Returns (ttGlyph, advance_width) or (None, None) if the stroke is degenerate.
    """
    line = LineString(points_xy)
    if line.length < 1e-6:
        return None, None

    poly = line.buffer(stroke_width_frac / 2, cap_style=1, join_style=1, resolution=8)
    if poly.is_empty:
        return None, None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)  # keep the dominant piece

    contours = _polygon_to_contours(poly)

    pen = TTGlyphPen(glyphSet=None)
    x_min = min(x for contour in contours for x, _ in contour)
    x_max = max(x for contour in contours for x, _ in contour)

    left_bearing = stroke_width_frac * SIDE_BEARING_FRAC
    for contour in contours:
        scaled = [
            (round((x - x_min + left_bearing) * units_per_em), round(y * units_per_em))
            for x, y in contour
        ]
        pen.moveTo(scaled[0])
        for pt in scaled[1:]:
            pen.lineTo(pt)
        pen.closePath()

    glyph = pen.glyph()
    advance_width = round((x_max - x_min + 2 * left_bearing) * units_per_em)
    return glyph, advance_width


def _notdef_glyph(units_per_em=UNITS_PER_EM):
    pen = TTGlyphPen(glyphSet=None)
    box = [(100, 0), (100, 700), (600, 700), (600, 0)]
    pen.moveTo(box[0])
    for pt in box[1:]:
        pen.lineTo(pt)
    pen.closePath()
    return pen.glyph(), 700


def _glyph_name(char):
    if char == " ":
        return "space"
    if char.isalnum() and char.isascii():
        return char
    return f"uni{ord(char):04X}"


def build_font(char_to_model, output_path, family_name, style_name="Regular",
                stroke_width_frac=0.06, num_points=400):
    """
    char_to_model: dict mapping a single character -> trained NeuralBasisCurve.
    Writes a .ttf to output_path.
    """
    glyph_order = [".notdef", "space"]
    glyphs = {}
    metrics = {}
    cmap = {}

    notdef_glyph, notdef_width = _notdef_glyph()
    glyphs[".notdef"] = notdef_glyph
    metrics[".notdef"] = (notdef_width, 0)

    space_width = round(UNITS_PER_EM * 0.4)
    glyphs["space"] = TTGlyphPen(glyphSet=None).glyph()
    metrics["space"] = (space_width, 0)
    cmap[ord(" ")] = "space"

    skipped = []
    for char, model in char_to_model.items():
        _, pts = eval_dense(model, num_points=num_points)
        glyph, advance_width = curve_to_glyph(pts, stroke_width_frac=stroke_width_frac)
        if glyph is None:
            skipped.append(char)
            continue
        name = _glyph_name(char)
        glyph_order.append(name)
        glyphs[name] = glyph
        metrics[name] = (advance_width, glyph.xMin if hasattr(glyph, "xMin") else 0)
        cmap[ord(char)] = name

    if skipped:
        print(f"[font_export] skipped degenerate glyphs: {skipped}")

    fb = FontBuilder(unitsPerEm=UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({"familyName": family_name, "styleName": style_name})
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT,
                usWinAscent=ASCENT, usWinDescent=abs(DESCENT))
    fb.setupPost()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(output_path))
    return output_path
