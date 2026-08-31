"""
Raster handwriting -> ordered, arc-length-parameterized curve.

Pipeline: grayscale image -> Otsu threshold -> skeletonize -> build an
8-connected pixel graph -> Eulerian trail (Hierholzer) gives a single
ordered walk over the stroke -> normalize and parameterize by arc length.

Adapted from the AM111 final project (github.com/tuechile/AM111_Final).
"""

from collections import deque

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


def _largest_connected_component(adj):
    """adj: adjacency list (index -> list of neighbor indices). Returns the
    vertex indices of the largest connected component, via BFS."""
    n = len(adj)
    visited = [False] * n
    best = []
    for seed in range(n):
        if visited[seed]:
            continue
        visited[seed] = True
        component = [seed]
        queue = [seed]
        while queue:
            v = queue.pop()
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    component.append(u)
                    queue.append(u)
        if len(component) > len(best):
            best = component
    return best


def _bfs_shortest_path(adj, start, end):
    """Shortest path (list of vertex indices, start..end inclusive) over the
    real pixel-adjacency graph. Used to patch phantom Hierholzer jumps -
    see extract_ordered_curve_from_skeleton."""
    if start == end:
        return [start]
    parent = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for u in adj[v]:
            if u not in parent:
                parent[u] = v
                if u == end:
                    path = [end]
                    while path[-1] is not None:
                        path.append(parent[path[-1]])
                    path.pop()  # drop the trailing None
                    path.reverse()
                    return path
                queue.append(u)
    raise ValueError("no path found - start and end should be in the same connected component")


def extract_ordered_curve_from_skeleton(skeleton):
    """
    skeleton: boolean array, True at stroke pixels.

    Builds an 8-connected pixel graph and extracts an Eulerian trail via
    Hierholzer's algorithm, giving one continuous ordered walk over the
    stroke (revisiting junction pixels as needed for self-intersections).

    Hierholzer's algorithm is only exact for a graph with 0 or 2 odd-degree
    vertices (a closed loop, or a single open stroke). A real 3-or-more-way
    junction - a crossbar meeting a stem, strokes crossing in cursive script
    - gives the skeleton graph more than 2 odd-degree vertices, and the
    stack-based algorithm below can then emit a "phantom" edge directly
    between two sibling branches of a junction that aren't actually
    adjacent (they only share the junction vertex). Left alone, that shows
    up as a straight line cutting across the traced curve where there
    should be a short backtrack through the junction instead. So after the
    initial trail is built, any consecutive pair that isn't a real pixel
    edge gets patched with the actual shortest path between them.

    If the skeleton has multiple disconnected components (stray noise, or
    a multi-stroke glyph like 'i'/'j' where the dot is separate from the
    stem), only the largest component is traced - a single Eulerian trail
    can't cross between disconnected pieces, so smaller components (e.g.
    an 'i' dot) are silently dropped rather than traced instead of the stem.

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

    largest_component = _largest_connected_component(adj)
    odd_vertices = [i for i in largest_component if len(adj[i]) % 2 == 1]
    start = odd_vertices[0] if odd_vertices else largest_component[0]

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

    patched_trail = [trail[0]]
    for a, b in zip(trail[:-1], trail[1:]):
        if b in adj[a]:
            patched_trail.append(b)
        else:
            patched_trail.extend(_bfs_shortest_path(adj, a, b)[1:])
    trail = patched_trail

    coords = coords_pix[trail]

    # Normalize by one shared scale (not independently per axis) so a thin
    # stroke's true proportions survive - independent min-max normalization
    # stretches a narrow glyph's few-pixel skeletonization wobble across the
    # full [0,1] range, turning sub-pixel noise into a large visual artifact
    # (most visible on tall/thin strokes like 'i', 'j', 'l').
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, 1e-8)
    coords[:, 0] = (coords[:, 0] - x_min) / span
    coords[:, 1] = (coords[:, 1] - y_min) / span
    coords[:, 1] = coords[:, 1].max() - coords[:, 1]  # flip y within its own extent (image row 0 is top, curves want y-up)

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
