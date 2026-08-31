"""
NeuralBasisCurve: a K-control-point curve with a learned neural basis.

    C(s) = softmax(MLP(s)) @ ctrl

MLP mixes a sin() layer (high-frequency capacity - needed so loops and
self-intersections don't get "shortcut" the way a pure-tanh network does)
with a tanh layer (smooths the basis so it doesn't overfit noise). K
controls how compact the representation is: this is the same K-point
mechanism explored in the AM111 final project.

Adapted from github.com/tuechile/AM111_Final (nofixedbasis_neural.py).
"""

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
    torch.save({"k": model.k, "state_dict": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    k = ckpt["k"]
    dummy_ctrl = np.zeros((k, 2), dtype=np.float32)
    model = NeuralBasisCurve(k, init_ctrl_points=dummy_ctrl)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
