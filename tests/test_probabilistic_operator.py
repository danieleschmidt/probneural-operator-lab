"""
Tests for probneural_operator.probabilistic_operator
=====================================================
Run with:
    ~/anaconda3/bin/python3 -m pytest tests/test_probabilistic_operator.py -v
"""
import math
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from probneural_operator.probabilistic_operator import (
    ProbabilisticFNO1D,
    LaplacePosterior,
    CalibrationEvaluator,
    ActiveLearningSelector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX = 16   # tiny grid for speed


def _make_toy_data(n: int = 30, nx: int = NX, seed: int = 42):
    """Simple linear map: output = 0.5 * input, for fast unit tests."""
    torch.manual_seed(seed)
    x = torch.randn(n, 1, nx)
    y = 0.5 * x + 0.1 * torch.randn(n, 1, nx)
    return x, y


@pytest.fixture
def tiny_model():
    return ProbabilisticFNO1D(nx=NX, modes=4, width=8, depth=2, prior_precision=1.0)


@pytest.fixture
def trained_model():
    """Lightly-trained model + fitted posterior."""
    torch.manual_seed(0)
    model = ProbabilisticFNO1D(nx=NX, modes=4, width=8, depth=2, prior_precision=1.0)
    x, y = _make_toy_data(40)
    loader = DataLoader(TensorDataset(x, y), batch_size=20, shuffle=False)
    model.fit(loader, epochs=5, lr=1e-3, device="cpu", verbose=False)
    # Use small subset for posterior to keep test fast
    post_loader = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=8, shuffle=False)
    model.fit_posterior(post_loader)
    return model


# ---------------------------------------------------------------------------
# ProbabilisticFNO1D
# ---------------------------------------------------------------------------

class TestProbabilisticFNO1D:
    def test_forward_shape(self, tiny_model):
        x = torch.randn(4, 1, NX)
        out = tiny_model(x)
        assert out.shape == (4, 1, NX), f"Expected (4,1,{NX}), got {out.shape}"

    def test_fit_reduces_loss(self, tiny_model):
        x, y = _make_toy_data(40)
        loader = DataLoader(TensorDataset(x, y), batch_size=20)
        history = tiny_model.fit(loader, epochs=10, lr=1e-3, device="cpu", verbose=False)
        assert history["train_loss"][-1] < history["train_loss"][0], \
            "Loss should decrease after 10 epochs"

    def test_predict_shape(self, trained_model):
        x = torch.randn(5, 1, NX)
        mean, std = trained_model.predict(x, return_std=True)
        assert mean.shape == (5, 1, NX)
        assert std.shape == (5, 1, NX)

    def test_predict_no_std(self, trained_model):
        x = torch.randn(3, 1, NX)
        mean, std = trained_model.predict(x, return_std=False)
        assert mean.shape == (3, 1, NX)
        assert std is None

    def test_std_positive(self, trained_model):
        x = torch.randn(4, 1, NX)
        _, std = trained_model.predict(x, return_std=True)
        assert (std >= 0).all(), "Standard deviation must be non-negative"

    def test_posterior_not_fitted_raises(self, tiny_model):
        x = torch.randn(2, 1, NX)
        with pytest.raises(RuntimeError, match="fit()"):
            tiny_model.predict(x, return_std=True)

    def test_parameter_count(self):
        model = ProbabilisticFNO1D(nx=NX, modes=4, width=16, depth=2)
        total = sum(p.numel() * (2 if p.is_complex() else 1)
                    for p in model.parameters() if p.requires_grad)
        assert total > 0


# ---------------------------------------------------------------------------
# LaplacePosterior
# ---------------------------------------------------------------------------

class TestLaplacePosterior:
    def test_fit_sets_precision(self):
        model = ProbabilisticFNO1D(nx=NX, modes=4, width=8, depth=2, prior_precision=1.0)
        x, y = _make_toy_data(10)
        loader = DataLoader(TensorDataset(x, y), batch_size=10)
        model.fit(loader, epochs=2, lr=1e-3, device="cpu", verbose=False)

        post = LaplacePosterior(model, prior_precision=1.0)
        post.fit(loader, torch.device("cpu"))

        assert post.diag_precision is not None
        assert post._is_fitted
        assert (post.diag_precision >= 1.0).all(), \
            "Posterior precision should be >= prior precision"

    def test_precision_positive(self):
        model = ProbabilisticFNO1D(nx=NX, modes=4, width=8, depth=2)
        x, y = _make_toy_data(8)
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        model.fit(loader, epochs=2, device="cpu", verbose=False)
        post = LaplacePosterior(model, prior_precision=0.5)
        post.fit(loader, torch.device("cpu"))
        assert (post.diag_precision > 0).all()


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------

class TestCalibrationEvaluator:
    def test_evaluate_returns_coverage(self, trained_model):
        x, y = _make_toy_data(10)
        ev = CalibrationEvaluator(trained_model)
        results = ev.evaluate(x, y, levels=[0.50, 0.90])
        assert "coverage@0.50" in results
        assert "coverage@0.90" in results
        assert "mean_mse" in results
        assert "mean_nll" in results

    def test_coverage_range(self, trained_model):
        x, y = _make_toy_data(12)
        ev = CalibrationEvaluator(trained_model)
        results = ev.evaluate(x, y, levels=[0.50, 0.90, 0.95])
        for k, v in results.items():
            if k.startswith("coverage@"):
                assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0, 1]"

    def test_mse_positive(self, trained_model):
        x, y = _make_toy_data(6)
        ev = CalibrationEvaluator(trained_model)
        results = ev.evaluate(x, y)
        assert results["mean_mse"] >= 0.0

    def test_90pct_coverage_passes(self, trained_model):
        """The 90% interval should cover at least 90% of test points."""
        x, y = _make_toy_data(20)
        ev = CalibrationEvaluator(trained_model)
        results = ev.evaluate(x, y, levels=[0.90])
        assert results["coverage@0.90"] >= 0.90, \
            f"90% coverage failed: {results['coverage@0.90']:.2%}"


# ---------------------------------------------------------------------------
# ActiveLearningSelector
# ---------------------------------------------------------------------------

class TestActiveLearningSelector:
    def test_select_returns_indices(self, trained_model):
        x_pool = torch.randn(20, 1, NX)
        sel = ActiveLearningSelector(trained_model, n_select=5)
        indices = sel.select(x_pool)
        assert indices.shape == (5,)
        assert indices.dtype == torch.int64

    def test_select_indices_unique(self, trained_model):
        x_pool = torch.randn(15, 1, NX)
        sel = ActiveLearningSelector(trained_model, n_select=5)
        indices = sel.select(x_pool)
        assert len(indices.unique()) == len(indices), "Selected indices must be unique"

    def test_select_n_capped_at_pool_size(self, trained_model):
        x_pool = torch.randn(3, 1, NX)
        sel = ActiveLearningSelector(trained_model, n_select=10)
        indices = sel.select(x_pool)
        assert len(indices) == 3, "Cannot select more points than pool size"

    def test_selected_above_average_uncertainty(self, trained_model):
        torch.manual_seed(5)
        x_pool = torch.randn(30, 1, NX)
        sel = ActiveLearningSelector(trained_model, n_select=5)
        scores = sel.uncertainty_scores(x_pool)
        indices = sel.select(x_pool)
        assert scores[indices].mean() >= scores.mean(), \
            "Selected points should have above-average uncertainty"

    def test_uncertainty_scores_shape(self, trained_model):
        x_pool = torch.randn(12, 1, NX)
        sel = ActiveLearningSelector(trained_model, n_select=3)
        scores = sel.uncertainty_scores(x_pool)
        assert scores.shape == (12,)
        assert (scores >= 0).all()
