"""Small, dependency-light perceptron implementation for Session 6."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PerceptronResult:
    weights: np.ndarray
    bias: float
    converged: bool
    epochs: int
    updates: int
    errors: list[int]
    updates_per_epoch: list[int]
    weight_history: list[np.ndarray]
    bias_history: list[float]


def predict(X, weights, bias):
    """Predict labels in {-1, +1}; a zero score belongs to class +1."""
    X = np.asarray(X, dtype=float)
    scores = X @ np.asarray(weights, dtype=float) + float(bias)
    return np.where(scores >= 0.0, 1, -1)


def accuracy(X, y, weights, bias):
    return float(np.mean(predict(X, weights, bias) == np.asarray(y)))


def train_perceptron(
    X,
    y,
    learning_rate=1.0,
    initial_weights=None,
    initial_bias=0.0,
    max_epochs=100,
    shuffle=False,
    random_state=0,
):
    """Train the classic online perceptron and record its full trajectory."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if y.shape != (len(X),):
        raise ValueError("y must contain exactly one label per sample.")
    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Perceptron labels must be -1 or +1.")
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must satisfy 0 < alpha <= 1.")
    if max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")

    weights = (
        np.zeros(X.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float).copy()
    )
    if weights.shape != (X.shape[1],):
        raise ValueError("initial_weights has the wrong dimension.")

    bias = float(initial_bias)
    rng = np.random.default_rng(random_state)
    errors = []
    updates_per_epoch = []
    weight_history = [weights.copy()]
    bias_history = [bias]
    updates = 0

    for epoch in range(1, max_epochs + 1):
        epoch_errors = 0
        order = rng.permutation(len(X)) if shuffle else np.arange(len(X))

        for idx in order:
            signed_score = y[idx] * (weights @ X[idx] + bias)
            if signed_score <= 0:
                weights += learning_rate * y[idx] * X[idx]
                bias += learning_rate * y[idx]
                updates += 1
                epoch_errors += 1
                weight_history.append(weights.copy())
                bias_history.append(bias)

        updates_per_epoch.append(epoch_errors)
        current_errors = int(np.sum(y * (X @ weights + bias) <= 0))
        errors.append(current_errors)
        if epoch_errors == 0:
            return PerceptronResult(
                weights,
                bias,
                True,
                epoch,
                updates,
                errors,
                updates_per_epoch,
                weight_history,
                bias_history,
            )

    return PerceptronResult(
        weights,
        bias,
        False,
        max_epochs,
        updates,
        errors,
        updates_per_epoch,
        weight_history,
        bias_history,
    )


def make_synthetic_data(kind="separable", n_per_class=30, random_state=7):
    """Create deterministic 2D examples for the comparison task."""
    rng = np.random.default_rng(random_state)
    if kind == "separable":
        negative = rng.normal(loc=(-2.0, -1.5), scale=0.55, size=(n_per_class, 2))
        positive = rng.normal(loc=(2.0, 1.5), scale=0.55, size=(n_per_class, 2))
    elif kind == "nonseparable":
        negative = rng.normal(loc=(0.0, 0.0), scale=1.25, size=(n_per_class, 2))
        positive = rng.normal(loc=(0.45, 0.35), scale=1.25, size=(n_per_class, 2))
    else:
        raise ValueError("kind must be 'separable' or 'nonseparable'.")

    X = np.vstack([negative, positive])
    y = np.hstack([-np.ones(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y
