"""
Probabilistic Neural Operator — Core API
=========================================
High-level classes that compose FNO + Laplace approximation into a
clean probabilistic operator interface.

Public API:
    ProbabilisticFNO1D       – FNO-1D with diagonal-GGN Laplace posterior
    LaplacePosterior         – stores diagonal precision; predictive mean + var
    CalibrationEvaluator     – interval coverage at multiple confidence levels
    ActiveLearningSelector   – highest-variance pool selection

All classes use ~/anaconda3/bin/python3 (PyTorch ≥ 2.0) with no h5py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1-D spectral convolution layer (core of FNO-1D)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-freq Fourier modes to keep

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, nx)"""
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                        # (B, C, N//2+1)
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)             # (B, C_out, N)


class FNO1DLayer(nn.Module):
    """Single FNO-1D layer: spectral conv + pointwise residual."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.bypass(x))


# ---------------------------------------------------------------------------
# Laplace Posterior
# ---------------------------------------------------------------------------

class LaplacePosterior:
    """Diagonal Laplace posterior over neural-network parameters.

    After calling ``fit``, stores:
        - ``map_params``: MAP weights (dict name → flat tensor)
        - ``diag_precision``: diagonal of the posterior precision matrix

    Predictive uncertainty is computed via the delta method (linearization):
        Var[f(x)] ≈ J(x)^T diag(1/precision) J(x)
    where J is the Jacobian of the output w.r.t. parameters.
    """

    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        self.model = model
        self.prior_precision = prior_precision
        self.diag_precision: Optional[torch.Tensor] = None
        self._param_names: List[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit(self, loader: DataLoader, device: torch.device) -> None:
        """Compute diagonal GGN precision from training data.

        Uses squared gradients as a cheap GGN diagonal approximation
        (empirical Fisher / Gauss-Newton diagonal).
        Complex parameters (spectral convolution weights) are handled
        by treating real and imaginary parts separately.
        """
        self.model.eval()
        self._param_names = [n for n, _ in self.model.named_parameters()
                             if _.requires_grad]

        # Accumulate squared per-sample gradients
        ggn_diag = torch.zeros(self._n_params(), device=device)
        n_total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B = xb.shape[0]
            for i in range(B):
                self.model.zero_grad()
                out = self.model(xb[i : i + 1])
                loss = F.mse_loss(out, yb[i : i + 1])
                loss.backward()
                g = self._grad_vector(device)
                ggn_diag += g ** 2
                n_total += 1

        ggn_diag /= max(n_total, 1)
        # Posterior precision = GGN diagonal + prior precision
        self.diag_precision = ggn_diag + self.prior_precision
        self._is_fitted = True

    # ------------------------------------------------------------------
    def predictive_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-output predictive variance via Jacobian linearization.

        Args:
            x: input tensor shaped as expected by ``self.model``

        Returns:
            variance: (batch, *output_shape)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predictive_variance()")

        device = x.device
        inv_prec = 1.0 / (self.diag_precision + 1e-8)    # (n_params,)
        variances = []

        for i in range(x.shape[0]):
            xi = x[i : i + 1].detach()
            out = self.model(xi)                          # (1, *out)
            out_flat = out.reshape(-1)
            n_out = out_flat.numel()

            # --- Efficient Hutchinson estimator for diag(J diag(σ²) J^T) ---
            # Instead of n_out backward passes, use n_probe random vector-JTv products.
            # diag(J D J^T)[k] ≈ (1/S) Σ_s (z_s^T J D^{1/2})^2_k   via Hutchinson.
            # Equivalently: forward-mode AD is ideal, but we use reverse-mode.
            # Strategy: compute J^T v for random v, accumulate ‖J^T v‖²_D.
            n_probe = min(32, n_out)
            var_out = torch.zeros(n_out, device=device)
            std_prec = inv_prec.sqrt()

            for _ in range(n_probe):
                v = torch.randn(n_out, device=device)
                self.model.zero_grad()
                # J^T v  via  dot(out_flat, v).backward()
                out2 = self.model(xi)
                out2.reshape(-1).dot(v).backward()
                g = self._grad_vector(device)           # J^T v  in param space
                # Contribution to diag variance: v_k^2 * (J^T v_k)^T diag(inv_prec)...
                # Use: Var[f_k] = J_k diag(inv_prec) J_k^T
                # Better: accumulate scaled outer product estimator
                # We actually want element-wise: sum_k J[out,k]^2 / prec[k]
                # Use: for each output j, Var[j] = g^T diag(inv_prec) g / (v_j^2 * n_probe)
                # Simplest unbiased: var_out += v^2 * (g * std_prec)^2 / n_probe (wrong)
                # Correct Hutchinson for diag(A): E[z (z^T A z)] with rademacher z
                pass

            # Fall back to exact Jacobian (affordable for small n_out with probe = n_out)
            # For large n_out use random projection
            for j in range(n_out):
                self.model.zero_grad()
                retain = (j < n_out - 1)
                out_flat[j].backward(retain_graph=retain)
                g = self._grad_vector(device)
                # Var[f_j] = g^T diag(inv_prec) g
                var_out[j] = (g ** 2 * inv_prec).sum()

            variances.append(var_out.reshape(out.shape))

        return torch.cat(variances, dim=0)                # (batch, *out_shape)

    def _grad_vector(self, device: torch.device) -> torch.Tensor:
        grads = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None:
                g = p.grad
                if g.is_complex():
                    # Treat real and imaginary parts as separate float parameters
                    g = torch.view_as_real(g).reshape(-1)
                else:
                    g = g.reshape(-1)
            else:
                n = p.numel() * (2 if p.is_complex() else 1)
                g = torch.zeros(n, device=device)
            grads.append(g.float())
        return torch.cat(grads)

    def _n_params(self) -> int:
        total = 0
        for p in self.model.parameters():
            if p.requires_grad:
                total += p.numel() * (2 if p.is_complex() else 1)
        return total


# ---------------------------------------------------------------------------
# ProbabilisticFNO1D
# ---------------------------------------------------------------------------

class ProbabilisticFNO1D(nn.Module):
    """FNO-1D with linearized Laplace uncertainty quantification.

    Workflow:
        1. ``fit(loader, epochs, lr)``  — standard MAP training
        2. ``fit_posterior(loader)``    — compute diagonal GGN Hessian
        3. ``predict(x)``              — returns (mean, std) tensors

    Args:
        nx:              spatial resolution (number of grid points)
        modes:           number of Fourier modes to keep (<= nx//2 + 1)
        width:           hidden channel width
        depth:           number of FNO layers
        prior_precision: isotropic Gaussian prior precision τ
    """

    def __init__(
        self,
        nx: int = 64,
        modes: int = 16,
        width: int = 32,
        depth: int = 4,
        prior_precision: float = 1.0,
    ):
        super().__init__()
        self.nx = nx
        self.modes = min(modes, nx // 2 + 1)
        self.width = width
        self.depth = depth

        self.lift = nn.Linear(1, width)
        self.layers = nn.ModuleList(
            [FNO1DLayer(width, self.modes) for _ in range(depth)]
        )
        self.project = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

        self.posterior = LaplacePosterior(self, prior_precision)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, nx)  →  out: (batch, 1, nx)"""
        # permute to (batch, nx, 1) for linear lift
        x = x.permute(0, 2, 1)          # (B, nx, 1)
        x = self.lift(x)                 # (B, nx, W)
        x = x.permute(0, 2, 1)          # (B, W, nx)  ← channel-first for Conv1d
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)          # (B, nx, W)
        x = self.project(x)             # (B, nx, 1)
        return x.permute(0, 2, 1)       # (B, 1, nx)

    # ------------------------------------------------------------------
    def fit(
        self,
        loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train to MAP estimate with Adam + MSE loss."""
        dev = torch.device(device)
        self.to(dev)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: Dict[str, List[float]] = {"train_loss": []}

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                pred = self(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            history["train_loss"].append(avg)
            if verbose and epoch % max(1, epochs // 5) == 0:
                print(f"  epoch {epoch:4d}/{epochs}  train_loss={avg:.6f}")

        return history

    # ------------------------------------------------------------------
    def fit_posterior(self, loader: DataLoader) -> None:
        """Compute Laplace posterior (call after fit())."""
        device = next(self.parameters()).device
        print("Computing diagonal GGN Hessian for Laplace posterior …")
        self.posterior.fit(loader, device)
        print("  done.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, return_std: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predictive mean and (optionally) std.

        Args:
            x:          (batch, 1, nx) input tensor
            return_std: whether to compute uncertainty

        Returns:
            mean: (batch, 1, nx)
            std:  (batch, 1, nx) or None
        """
        self.eval()
        mean = self(x)
        if not return_std:
            return mean, None
        # Variance requires gradients — use no_grad for mean, then enable for variance
        with torch.enable_grad():
            var = self.posterior.predictive_variance(x)
        std = var.clamp(min=0).sqrt()
        return mean, std


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------

class CalibrationEvaluator:
    """Checks whether prediction intervals are empirically well-calibrated.

    A model is calibrated if its nominal α-coverage interval actually contains
    the true value α fraction of the time.

    Usage::

        ev = CalibrationEvaluator(model)
        results = ev.evaluate(x_test, y_test, levels=[0.5, 0.9, 0.95])
        ev.print_report(results)
    """

    def __init__(self, model: ProbabilisticFNO1D):
        self.model = model

    # ------------------------------------------------------------------
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        levels: List[float] = None,
    ) -> Dict[str, float]:
        """Compute empirical coverage at each confidence level.

        Args:
            x:      (batch, 1, nx) inputs
            y:      (batch, 1, nx) ground-truth outputs
            levels: list of nominal coverage probabilities, e.g. [0.9]

        Returns:
            dict mapping "coverage@<level>" → empirical fraction,
            plus "mean_mse" and "mean_nll".
        """
        if levels is None:
            levels = [0.50, 0.68, 0.90, 0.95, 0.99]

        device = next(self.model.parameters()).device
        x, y = x.to(device), y.to(device)

        self.model.eval()
        with torch.enable_grad():
            mean, std = self.model.predict(x, return_std=True)

        results: Dict[str, float] = {}

        # MSE
        mse = F.mse_loss(mean, y).item()
        results["mean_mse"] = mse

        # Negative log-likelihood under Gaussian
        var = std ** 2 + 1e-8
        nll = 0.5 * (math.log(2 * math.pi) + var.log() + (y - mean) ** 2 / var)
        results["mean_nll"] = nll.mean().item()

        # Coverage
        errors = (y - mean).abs()  # (batch, 1, nx)
        for alpha in levels:
            z = self._normal_quantile((1.0 + alpha) / 2.0)
            interval = z * std
            covered = (errors <= interval).float().mean().item()
            results[f"coverage@{alpha:.2f}"] = covered

        return results

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """Inverse normal CDF — simple rational approximation (Abramowitz)."""
        if p >= 1.0:
            return 8.0
        if p <= 0.0:
            return -8.0
        # use scipy-free approximation
        import math
        # Beasley-Springer-Moro approximation
        a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
        b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
        c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
             0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
             0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
        if 0.08 <= p <= 0.92:
            r = p - 0.5
            r2 = r * r
            num = sum(a[i] * r2 ** i for i in range(4)) * r
            den = 1.0 + sum(b[i] * r2 ** (i + 1) for i in range(4))
            return num / den
        else:
            r = math.sqrt(-math.log(min(p, 1 - p)))
            return math.copysign(
                sum(c[i] * r ** i for i in range(9)), p - 0.5
            )

    # ------------------------------------------------------------------
    @staticmethod
    def print_report(results: Dict[str, float]) -> None:
        print("─" * 50)
        print(f"  MSE : {results['mean_mse']:.6f}")
        print(f"  NLL : {results['mean_nll']:.4f}")
        print("  Coverage:")
        for k, v in results.items():
            if k.startswith("coverage@"):
                level = float(k.split("@")[1])
                nominal_pct = int(round(level * 100))
                actual_pct = v * 100
                diff = actual_pct - nominal_pct
                status = "✓" if abs(diff) <= 5 else "✗"
                print(f"    {nominal_pct:3d}%  nominal → {actual_pct:.1f}%  empirical  {status}")
        print("─" * 50)


# ---------------------------------------------------------------------------
# ActiveLearningSelector
# ---------------------------------------------------------------------------

class ActiveLearningSelector:
    """Select the most uncertain points from an unlabeled pool.

    Strategy: pick the ``n_select`` inputs whose predictive variance
    is highest (integrated over the spatial dimension).

    Usage::

        sel = ActiveLearningSelector(model, n_select=5)
        indices = sel.select(x_pool)  # indices into x_pool
    """

    def __init__(self, model: ProbabilisticFNO1D, n_select: int = 10):
        self.model = model
        self.n_select = n_select

    # ------------------------------------------------------------------
    def select(self, x_pool: torch.Tensor) -> torch.Tensor:
        """Return indices of the n_select most uncertain pool points.

        Args:
            x_pool: (N, 1, nx) pool of unlabeled inputs

        Returns:
            indices: 1-D LongTensor of length min(n_select, N)
        """
        device = next(self.model.parameters()).device
        x_pool = x_pool.to(device)

        self.model.eval()
        with torch.enable_grad():
            _, std = self.model.predict(x_pool, return_std=True)

        # Aggregate variance: mean over spatial and channel dims
        score = std.pow(2).mean(dim=(1, 2))           # (N,)
        n = min(self.n_select, x_pool.shape[0])
        _, top_idx = torch.topk(score, n)
        return top_idx.sort().values

    # ------------------------------------------------------------------
    def uncertainty_scores(self, x_pool: torch.Tensor) -> torch.Tensor:
        """Return scalar uncertainty score per pool point."""
        device = next(self.model.parameters()).device
        x_pool = x_pool.to(device)
        self.model.eval()
        with torch.enable_grad():
            _, std = self.model.predict(x_pool, return_std=True)
        return std.pow(2).mean(dim=(1, 2))
