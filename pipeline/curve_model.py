"""
Two K-control-point curve fits: a cubic B-spline (fixed basis) and a
NeuralBasisCurve (learned basis).

    C(s) = basis(s) @ ctrl

BSplineCurve uses a fixed Cox-de Boor basis; only the K control points are
learned, which makes fitting a near-linear least-squares problem for Adam.
NeuralBasisCurve instead learns the basis itself via a sin+tanh MLP - the
original hypothesis (AM111 final project) was that a learned basis would
handle self-intersecting loops (cursive e/o/l) better than a fixed one.

Benchmarked against each other on a self-intersecting stroke (see
tests/test_curve_model.py and the project chat history): B-spline won by
1-3 orders of magnitude in MSE at every K tested, and - contrary to the
original hypothesis - it was the NeuralBasisCurve fit that "shortcut"
across the self-intersection, while Bezier/B-spline traced it correctly.
BSplineCurve is what scripts/03_train_glyphs.py uses; NeuralBasisCurve is
kept for comparison/reference.

Adapted from github.com/tuechile/AM111_Final (nofixedbasis_neural.py,
3_mixed_methods/curve_neural.py).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SinLayer(nn.Module):
    def forward(self, x):
        return torch.sin(np.pi * x)


class NeuralBasisCurve(nn.Module):
    def __init__(self, k, init_ctrl_points):
        super().__init__()
        self.k = k
        self.ctrl = nn.Parameter(torch.tensor(init_ctrl_points, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            SinLayer(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, k),
        )

    def forward(self, t):
        t_in = t.view(-1, 1)
        logits = self.mlp(t_in)
        basis = torch.softmax(logits, dim=1)
        return basis @ self.ctrl


def _basis_smoothness_penalty(model, t):
    t_sorted, _ = torch.sort(t)
    logits = model.mlp(t_sorted.view(-1, 1))
    basis = torch.softmax(logits, dim=1)
    d_basis = basis[1:] - basis[:-1]
    return (d_basis ** 2).mean()


def _basis_locality_term(model, t):
    logits = model.mlp(t.view(-1, 1))
    basis = torch.softmax(logits, dim=1)
    entropy = -(basis * torch.log(basis + 1e-8)).sum(dim=1).mean()
    return -entropy


def _ctrl_smoothness_penalty(ctrl):
    if ctrl.shape[0] < 3:
        return torch.tensor(0.0, dtype=ctrl.dtype, device=ctrl.device)
    second_diff = ctrl[2:] - 2 * ctrl[1:-1] + ctrl[:-2]
    return (second_diff ** 2).mean()


def train_neural_basis(
    s,
    coords,
    k,
    num_epochs=3000,
    lr=1e-3,
    lam_basis_smooth=1e-3,
    lam_basis_local=1e-3,
    lam_ctrl_smooth=1e-2,
    verbose=False,
):
    """
    s: (N,) arc-length parameter in [0,1].
    coords: (N,2) target curve points.
    Returns (model, final_mse).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_train = torch.tensor(s, dtype=torch.float32, device=device)
    pts = torch.tensor(coords, dtype=torch.float32, device=device)

    n = len(coords)
    init_idx = np.linspace(0, n - 1, k, dtype=int)
    init_ctrl = coords[init_idx]

    model = NeuralBasisCurve(k, init_ctrl_points=init_ctrl).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(t_train)
        loss = criterion(pred, pts)
        if lam_basis_smooth > 0.0:
            loss = loss + lam_basis_smooth * _basis_smoothness_penalty(model, t_train)
        if lam_basis_local > 0.0:
            loss = loss - lam_basis_local * _basis_locality_term(model, t_train)
        if lam_ctrl_smooth > 0.0:
            loss = loss + lam_ctrl_smooth * _ctrl_smoothness_penalty(model.ctrl)

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  [K={k}] epoch {epoch + 1}/{num_epochs} loss={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred_final = model(t_train).cpu().numpy()
    mse_final = float(np.mean((pred_final - coords) ** 2))
    return model, mse_final


def _open_uniform_knots(n_ctrl, degree):
    """Open uniform (clamped) knot vector on [0,1]."""
    p = degree
    knots = np.zeros(n_ctrl + p + 1, dtype=np.float64)
    knots[-(p + 1):] = 1.0
    n_internal = n_ctrl - p - 1
    if n_internal > 0:
        knots[p + 1:p + 1 + n_internal] = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return knots


def _bspline_basis_at_t(t, n_ctrl, degree, knots):
    """N_{i,degree}(t) for i=0..n_ctrl-1 via Cox-de Boor recursion."""
    p = degree
    eps = 1e-8
    t = float(np.clip(t, knots[0] + eps, knots[-1] - eps))

    basis = np.array([1.0 if knots[i] <= t < knots[i + 1] else 0.0 for i in range(n_ctrl)])
    for k in range(1, p + 1):
        new_basis = np.zeros(n_ctrl, dtype=np.float64)
        for i in range(n_ctrl):
            denom1 = knots[i + k] - knots[i]
            term1 = (t - knots[i]) / denom1 * basis[i] if denom1 != 0.0 else 0.0
            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = (
                (knots[i + k + 1] - t) / denom2 * basis[i + 1]
                if denom2 != 0.0 and i + 1 < n_ctrl else 0.0
            )
            new_basis[i] = term1 + term2
        basis = new_basis
    return basis


def bspline_basis_matrix(t, n_ctrl, degree):
    """t: (N,) in [0,1]. Returns (N, n_ctrl) with B[i, j] = N_{j,degree}(t_i)."""
    t = np.asarray(t, dtype=np.float64)
    knots = _open_uniform_knots(n_ctrl, degree)
    basis = np.stack([_bspline_basis_at_t(ti, n_ctrl, degree, knots) for ti in t])
    return basis.astype(np.float32)


class BSplineCurve(nn.Module):
    """C(t) = B(t) @ ctrl, with a fixed (non-learned) B-spline basis B(t)."""

    def __init__(self, k, init_ctrl_points, degree=3):
        super().__init__()
        self.k = k
        self.degree = degree
        self.ctrl = nn.Parameter(torch.tensor(init_ctrl_points, dtype=torch.float32))

    def forward(self, t):
        t_np = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        basis = bspline_basis_matrix(t_np, self.k, self.degree)
        basis = torch.tensor(basis, dtype=torch.float32, device=self.ctrl.device)
        return basis @ self.ctrl


def train_bspline(s, coords, k, degree=3, num_epochs=2000, lr=1e-2, verbose=False):
    """
    s: (N,) arc-length parameter in [0,1].
    coords: (N,2) target curve points.
    Returns (model, final_mse).
    """
    if k <= degree:
        raise ValueError(f"k={k} must be > degree={degree} for a B-spline fit.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis_train = torch.tensor(bspline_basis_matrix(s, k, degree), dtype=torch.float32, device=device)
    pts = torch.tensor(coords, dtype=torch.float32, device=device)

    n = len(coords)
    init_idx = np.linspace(0, n - 1, k, dtype=int)
    init_ctrl = coords[init_idx]

    model = BSplineCurve(k, init_ctrl_points=init_ctrl, degree=degree).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = basis_train @ model.ctrl
        loss = criterion(pred, pts)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  [BSpline k={k}] epoch {epoch + 1}/{num_epochs} loss={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred_final = (basis_train @ model.ctrl).cpu().numpy()
    mse_final = float(np.mean((pred_final - coords) ** 2))
    return model, mse_final


def eval_dense(model, num_points=400):
    """Evaluate the trained curve on a dense, evenly spaced grid in [0,1]."""
    device = next(model.parameters()).device
    t_dense = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    t_torch = torch.tensor(t_dense, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pts = model(t_torch).cpu().numpy()
    return t_dense, pts


def save_model(model, path):
    if isinstance(model, BSplineCurve):
        torch.save(
            {"kind": "bspline", "k": model.k, "degree": model.degree, "state_dict": model.state_dict()},
            path,
        )
    else:
        torch.save({"kind": "neuralbasis", "k": model.k, "state_dict": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    k = ckpt["k"]
    kind = ckpt.get("kind", "neuralbasis")  # older checkpoints predate the "kind" tag
    dummy_ctrl = np.zeros((k, 2), dtype=np.float32)
    if kind == "bspline":
        model = BSplineCurve(k, init_ctrl_points=dummy_ctrl, degree=ckpt["degree"])
    else:
        model = NeuralBasisCurve(k, init_ctrl_points=dummy_ctrl)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
