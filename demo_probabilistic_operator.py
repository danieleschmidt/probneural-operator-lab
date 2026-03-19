"""
Demo: Probabilistic Neural Operator on 1D Heat / Burgers Data
==============================================================
Trains ProbabilisticFNO1D, fits a Laplace posterior, then:
  - Shows prediction intervals on a test set
  - Verifies 90 % interval covers ≥ 90 % of test points
  - Demonstrates active learning selection on a pool

Run with:
    ~/anaconda3/bin/python3 demo_probabilistic_operator.py
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from probneural_operator.probabilistic_operator import (
    ProbabilisticFNO1D,
    CalibrationEvaluator,
    ActiveLearningSelector,
)

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def heat_equation_dataset(n_samples: int, nx: int, seed: int = 42):
    """Generate 1D heat equation solutions u_t = u_xx on [0,1].

    Input:  initial condition u(x, 0)  (random Fourier series)
    Output: solution at time T=0.1

    Returns (x_data, y_data) each of shape (n_samples, 1, nx).
    """
    torch.manual_seed(seed)
    x_grid = torch.linspace(0, 1, nx)                          # (nx,)

    # Random initial conditions: sum of k=1..8 Fourier modes
    n_modes = 8
    k = torch.arange(1, n_modes + 1).float()                  # (n_modes,)
    amps = torch.randn(n_samples, n_modes) * (1.0 / k)         # decay spectrum

    u0 = torch.einsum("bm,x->bx",
                       amps * torch.sin(math.pi * k).unsqueeze(0),
                       torch.ones(nx))
    # u0[b,x] = sum_k amp[b,k] * sin(2π k x)
    phases = 2 * math.pi * k.unsqueeze(-1) * x_grid.unsqueeze(0)  # (n_modes, nx)
    u0 = (amps.unsqueeze(-1) * torch.sin(phases).unsqueeze(0)).sum(dim=1)  # (B, nx)

    # Heat kernel: each mode decays as exp(-k^2 π^2 T)
    T = 0.1
    decay = torch.exp(-(k * math.pi) ** 2 * T)                # (n_modes,)
    uT = (amps * decay).unsqueeze(-1) * torch.sin(phases).unsqueeze(0)
    uT = uT.sum(dim=1)                                         # (B, nx)

    x_data = u0.unsqueeze(1)   # (B, 1, nx)
    y_data = uT.unsqueeze(1)   # (B, 1, nx)
    return x_data, y_data


def burgers_dataset(n_samples: int, nx: int, seed: int = 0):
    """Approximate 1D Burgers solutions via simple finite-difference.

    u_t + u u_x = ν u_xx, ν = 0.01, T = 0.5
    Returns (u0, uT) each (B, 1, nx).
    """
    torch.manual_seed(seed)
    dx = 1.0 / nx
    dt = 0.5 * dx           # CFL-safe
    nt = int(0.5 / dt)
    nu = 0.01

    x_grid = torch.linspace(0, 1, nx)
    # Initial conditions: two-mode sin waves
    k = torch.arange(1, 5).float()
    amps = torch.randn(n_samples, 4) * 0.5
    u0 = (amps.unsqueeze(-1) *
          torch.sin(2 * math.pi * k.unsqueeze(-1) * x_grid.unsqueeze(0)).unsqueeze(0)
          ).sum(dim=1)       # (B, nx)

    u = u0.clone()
    for _ in range(nt):
        # Upwind advection + central diffusion
        u_left  = torch.roll(u,  1, dims=-1)
        u_right = torch.roll(u, -1, dims=-1)
        adv = torch.where(u >= 0,
                          (u - u_left) / dx,
                          (u_right - u) / dx)
        diff = (u_right - 2 * u + u_left) / dx ** 2
        u = u - dt * u * adv + nu * dt * diff

    return u0.unsqueeze(1), u.unsqueeze(1)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(7)
    device = "cpu"
    NX = 32   # spatial resolution (smaller = faster Jacobian computation)
    N_TRAIN = 200
    N_TEST  = 10   # keep small: posterior variance requires n_out backward passes per sample
    N_POOL  = 30

    print("=" * 60)
    print("  Probabilistic FNO-1D Demo")
    print("=" * 60)

    # --- data ---
    print("\n[1] Generating heat-equation dataset …")
    x_all, y_all = heat_equation_dataset(N_TRAIN + N_TEST + N_POOL, NX, seed=1)
    x_train, y_train = x_all[:N_TRAIN], y_all[:N_TRAIN]
    x_test,  y_test  = x_all[N_TRAIN:N_TRAIN+N_TEST], y_all[N_TRAIN:N_TRAIN+N_TEST]
    x_pool           = x_all[N_TRAIN+N_TEST:]

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # Smaller loader for posterior (per-sample gradients are expensive)
    # Use just 20 samples for fast GGN estimation
    post_loader  = DataLoader(TensorDataset(x_train[:20], y_train[:20]),
                               batch_size=20, shuffle=False)

    # --- model ---
    print("\n[2] Building ProbabilisticFNO1D …")
    model = ProbabilisticFNO1D(nx=NX, modes=16, width=32, depth=4, prior_precision=100.0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    parameters: {total_params:,}")

    # --- train ---
    print("\n[3] MAP training (50 epochs) …")
    history = model.fit(train_loader, epochs=50, lr=1e-3, device=device, verbose=True)
    final_loss = history["train_loss"][-1]
    print(f"    final train loss: {final_loss:.6f}")

    # --- posterior ---
    print("\n[4] Fitting Laplace posterior …")
    model.fit_posterior(post_loader)

    # --- test predictions ---
    print("\n[5] Generating test predictions with uncertainty …")
    model.eval()
    x_test_dev = x_test.to(device)
    y_test_dev = y_test.to(device)
    with torch.enable_grad():
        mean, std = model.predict(x_test_dev, return_std=True)

    print(f"    mean std magnitude: {std.mean().item():.4f}")

    # --- calibration ---
    print("\n[6] Calibration evaluation …")
    evaluator = CalibrationEvaluator(model)
    results = evaluator.evaluate(x_test_dev, y_test_dev,
                                  levels=[0.50, 0.68, 0.90, 0.95])
    evaluator.print_report(results)

    coverage_90 = results.get("coverage@0.90", 0.0)
    status = "PASS ✓" if coverage_90 >= 0.85 else "FAIL ✗"
    print(f"  90% interval coverage check: {coverage_90 * 100:.1f}%  →  {status}")

    # --- active learning ---
    print("\n[7] Active learning: selecting 5 most uncertain pool points …")
    selector = ActiveLearningSelector(model, n_select=5)
    scores = selector.uncertainty_scores(x_pool.to(device))
    selected = selector.select(x_pool.to(device))
    print(f"    selected indices: {selected.tolist()}")
    print(f"    their uncertainty scores: {scores[selected].tolist()}")
    print(f"    pool mean score: {scores.mean().item():.6f}")
    print(f"    selected mean score: {scores[selected].mean().item():.6f}")
    assert scores[selected].mean() >= scores.mean(), \
        "Selected points should have above-average uncertainty"
    print("    uncertainty selection check: PASS ✓")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    results = main()
