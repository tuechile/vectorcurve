"""
Raster handwriting -> ordered, arc-length-parameterized curve.

Pipeline: grayscale image -> Otsu threshold -> skeletonize -> build an
8-connected pixel graph -> Eulerian trail (Hierholzer) gives a single
ordered walk over the stroke -> normalize and parameterize by arc length.

Adapted from the AM111 final project (github.com/tuechile/AM111_Final).
"""

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


def load_and_skeletonize(img_gray):
    """
    img_gray: 2D float array in [0,1], grayscale, dark ink on light background.
    Returns a boolean skeleton array (True at 1-pixel-wide stroke centerline).
    """
    thresh = threshold_otsu(img_gray)
    binary = img_gray < thresh
    if not binary.any():
        raise ValueError("Thresholding found no ink pixels - check the input image.")
    return skeletonize(binary)


def extract_ordered_curve_from_skeleton(skeleton):
    """
    skeleton: boolean array, True at stroke pixels.

    Builds an 8-connected pixel graph and extracts an Eulerian trail via
    Hierholzer's algorithm, giving one continuous ordered walk over the
    stroke (revisiting junction pixels as needed for self-intersections).

    Returns coords_ordered: (M, 2) float array, normalized to [0,1]^2 with
    y flipped so it matches standard "y-up" curve orientation.
    """
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("No skeleton pixels found - check input image / thresholding.")

    n = xs.size
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)
    coord_to_idx = {(int(ys[i]), int(xs[i])): i for i in range(n)}

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),           (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]

    adj = [[] for _ in range(n)]
    for i in range(n):
        r, c = int(ys[i]), int(xs[i])
        for dr, dc in neighbor_offsets:
            j = coord_to_idx.get((r + dr, c + dc))
            if j is not None:
                adj[i].append(j)

    odd_vertices = [i for i in range(n) if len(adj[i]) % 2 == 1]
    start = odd_vertices[0] if odd_vertices else 0

    adj_copy = [nbrs.copy() for nbrs in adj]
    stack = [start]
    trail = []
    while stack:
        v = stack[-1]
        if adj_copy[v]:
            u = adj_copy[v].pop()
            try:
                adj_copy[u].remove(v)
            except ValueError:
                pass
            stack.append(u)
        else:
            trail.append(stack.pop())
    trail.reverse()

    coords = coords_pix[trail]

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min + 1e-8)
    coords[:, 1] = 1.0 - coords[:, 1]  # flip y: image row 0 is top, curves want y-up

    return coords


def parameterize_by_arclength(coords):
    """
    coords: (N, 2) ordered points.
    Returns s: (N,) arc-length parameter in [0,1], monotonically increasing.
    """
    diffs = coords[1:] - coords[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s = arc / (arc[-1] + 1e-8)
    return s


def image_to_curve(img_gray):
    """Convenience wrapper: grayscale image -> (s, coords, skeleton)."""
    skeleton = load_and_skeletonize(img_gray)
    coords = extract_ordered_curve_from_skeleton(skeleton)
    s = parameterize_by_arclength(coords)
    return s, coords, skeleton
