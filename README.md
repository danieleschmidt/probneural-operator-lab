# probneural-operator-lab

**Probabilistic Neural Operators** — FNO/DeepONet extended with uncertainty quantification via linearized Laplace approximation.

Neural operators learn mappings between infinite-dimensional function spaces (great for PDEs). This repo adds calibrated predictive uncertainty on top, enabling:

- **Confidence intervals** on operator predictions
- **Active learning** — query the most uncertain points in a pool
- **Calibration evaluation** — check that 90% intervals really cover 90% of test data

---

## Core Abstractions

| Class | What it does |
|---|---|
| `ProbabilisticFNO1D` | FNO-1D backbone. Train normally (`fit`), then call `fit_posterior` to compute the Laplace posterior. `predict(x)` returns `(mean, std)`. |
| `LaplacePosterior` | Stores diagonal GGN precision. Predictive variance via Jacobian linearization: `Var[f(x)] ≈ J(x)^T diag(1/precision) J(x)`. |
| `CalibrationEvaluator` | Empirical coverage at multiple confidence levels. Prints a calibration report. |
| `ActiveLearningSelector` | Picks the highest-variance pool points for labeling. |

---

## Quick Start

```python
from torch.utils.data import DataLoader, TensorDataset
from probneural_operator.probabilistic_operator import (
    ProbabilisticFNO1D, CalibrationEvaluator, ActiveLearningSelector,
)

# --- 1. Build model ---
model = ProbabilisticFNO1D(nx=64, modes=16, width=32, depth=4, prior_precision=100.0)

# --- 2. MAP training ---
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
model.fit(loader, epochs=50, lr=1e-3)

# --- 3. Laplace posterior ---
post_loader = DataLoader(TensorDataset(x_train[:50], y_train[:50]), batch_size=50)
model.fit_posterior(post_loader)

# --- 4. Predict with uncertainty ---
mean, std = model.predict(x_test)  # (batch, 1, nx) each

# --- 5. Check calibration ---
ev = CalibrationEvaluator(model)
results = ev.evaluate(x_test, y_test, levels=[0.50, 0.90, 0.95])
ev.print_report(results)

# --- 6. Active learning ---
sel = ActiveLearningSelector(model, n_select=10)
idx = sel.select(x_pool)   # indices of 10 most uncertain pool points
```

---

## Demo

```bash
~/anaconda3/bin/python3 demo_probabilistic_operator.py
```

Trains on 1D heat-equation data (random Fourier initial conditions → solution at T=0.1), verifies 90% prediction interval covers ≥90% of test points, and demonstrates active learning on a pool.

---

## Method

### Fourier Neural Operator (FNO-1D)

The FNO learns an operator $\mathcal{G}_\theta: \mathcal{A} \to \mathcal{B}$ between function spaces. Each layer applies a spectral convolution in Fourier space plus a pointwise bypass:

$$v_{t+1}(x) = \sigma\!\left(\mathcal{F}^{-1}[R_\theta \cdot \mathcal{F}(v_t)](x) + W v_t(x)\right)$$

where $R_\theta$ are learnable weights in Fourier space (only the first `modes` frequency components are kept).

### Diagonal Laplace Approximation

After MAP training, we approximate the posterior as:

$$p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\hat\theta_{\text{MAP}},\ \mathbf{H}^{-1})$$

where $\mathbf{H}$ is the generalized Gauss-Newton Hessian, approximated by its diagonal (empirical Fisher):

$$H_{ii} \approx \frac{1}{N} \sum_{n=1}^N \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2 + \tau$$

with $\tau$ the prior precision.

### Predictive Uncertainty

Via the delta method (first-order linearization around MAP):

$$\text{Var}[f_j(x)] \approx \mathbf{J}_j(x)^\top \mathbf{H}^{-1} \mathbf{J}_j(x)$$

where $\mathbf{J}_j$ is the Jacobian of output $j$ w.r.t. parameters. With diagonal $\mathbf{H}^{-1}$:

$$\text{Var}[f_j(x)] = \sum_i J_{ji}(x)^2 / H_{ii}$$

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/test_probabilistic_operator.py -v
```

18 tests covering forward pass shapes, training convergence, posterior fitting, calibration coverage, and active learning selection.

---

## Requirements

- PyTorch ≥ 2.0
- numpy
- No h5py, no heavy optional deps

```bash
pip install torch numpy pytest
```

---

## References

- Li et al. (2021). [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895). *ICLR 2021*.
- Daxberger et al. (2021). [Laplace Redux — Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806). *NeurIPS 2021*.
- MacKay (1992). A Practical Bayesian Framework for Backprop Networks. *Neural Computation*.
