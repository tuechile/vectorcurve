"""
Byproduct of the main pipeline, not the point of the project.

Two trained NeuralBasisCurve models for the "same" glyph (e.g. your 'a'
and a friend's 'a', or the same letter at two points in time) share an
identical architecture, so their weights can be linearly interpolated
directly. Because both curves are continuous functions of the same
arc-length parameter s, this produces a smooth in-between stroke rather
than a naive pixel cross-fade.
"""

import copy
import io

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .curve_model import eval_dense


def interpolate_models(model_a, model_b, alpha):
    if model_a.k != model_b.k:
        raise ValueError(
            f"Cannot morph curves with different K (control point counts): "
            f"{model_a.k} vs {model_b.k}. Retrain both with the same K."
        )
    model_out = copy.deepcopy(model_a)
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    new_sd = {key: (1 - alpha) * sd_a[key] + alpha * sd_b[key] for key in sd_a}
    model_out.load_state_dict(new_sd)
    model_out.eval()
    return model_out


def _render_frame(points_xy):
    fig = plt.figure(figsize=(3, 3), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(points_xy[:, 0], points_xy[:, 1], "k-", linewidth=2.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_morph_gif(model_a, model_b, output_path, n_frames=30, num_points=400,
                      hold_ends=4, frame_duration_ms=60):
    """
    Renders an animated GIF morphing model_a's curve into model_b's curve
    and back, holding briefly on each end.
    """
    alphas = np.linspace(0.0, 1.0, n_frames)
    frames = []
    for alpha in alphas:
        model = interpolate_models(model_a, model_b, float(alpha))
        _, pts = eval_dense(model, num_points=num_points)
        frames.append(_render_frame(pts))

    sequence = [frames[0]] * hold_ends + frames + [frames[-1]] * hold_ends + frames[::-1]

    sequence[0].save(
        output_path,
        save_all=True,
        append_images=sequence[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return output_path
