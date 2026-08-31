"""
Generates a printable SVG capture sheet: a grid of boxes, one per glyph,
sized in millimeters so it prints true-to-scale. Print it, write each
letter inside its box with a pen, then scan the page at a known DPI
(see scripts/02_segment_scan.py for the matching crop step).

v1 assumes a flatbed scan (or a very carefully aligned photo) - there is
no perspective correction here. Skew/perspective correction from a
casual phone photo is a reasonable follow-up, not in scope for v1.
"""

DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz"


def generate_template(charset=DEFAULT_CHARSET, cols=6, cell_mm=30, margin_mm=15, out_path="template.svg"):
    charset = list(charset)
    rows = -(-len(charset) // cols)  # ceil div

    width_mm = margin_mm * 2 + cols * cell_mm
    height_mm = margin_mm * 2 + rows * cell_mm

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {width_mm} {height_mm}">',
        f'<rect width="{width_mm}" height="{height_mm}" fill="white"/>',
    ]

    for i, char in enumerate(charset):
        row, col = divmod(i, cols)
        x = margin_mm + col * cell_mm
        y = margin_mm + row * cell_mm
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_mm}" height="{cell_mm}" '
            f'fill="none" stroke="black" stroke-width="0.2"/>'
        )
        parts.append(
            f'<text x="{x + 2}" y="{y + cell_mm - 2}" font-family="Georgia, serif" '
            f'font-size="{cell_mm * 0.6}" fill="#dddddd">{char}</text>'
        )

    parts.append("</svg>")
    svg = "\n".join(parts)

    with open(out_path, "w") as f:
        f.write(svg)

    return {
        "out_path": out_path,
        "charset": charset,
        "cols": cols,
        "rows": rows,
        "cell_mm": cell_mm,
        "margin_mm": margin_mm,
    }
