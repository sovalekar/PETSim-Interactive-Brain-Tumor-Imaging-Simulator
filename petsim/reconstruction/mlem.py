"""Maximum-likelihood expectation maximization (MLEM) for 2-D PET.

Update (standard emission MLEM)
--------------------------------
``x^{k+1}_j = (x^k_j / s_j) * Σ_i A_ij * y_i / (Σ_l A_il x^k_l)``

where ``s_j = Σ_i A_ij`` is the sensitivity of pixel ``j``.

``A`` is the parallel-beam projector in :mod:`petsim.reconstruction.projectors`
(forward) and its adjoint (backprojection). ``y`` is the measured sinogram.
The algorithm does **not** use phantom or tumor coordinates.

Assumptions
-----------
* Poisson counts in the binned parallel-beam sinogram.
* No attenuation, scatter, randoms, or detector-efficiency weights beyond the
  geometric projector (those effects are also absent from acquisition).
* MLEM is applied to the **unsmoothed** measured sinogram. FBP may still
  pre-smooth; that smoothing is not part of the MLEM likelihood.
* Pixels with zero sensitivity are left at 0 and are not updated.
* Bins with predicted counts ``ŷ_i = 0`` contribute 0 to the correction.
  If such a bin also has ``y_i > 0``, the Poisson log-likelihood is undefined
  and is reported as ``-inf``.
* The initial estimate is 1 on the sensitivity support and 0 elsewhere, so
  the multiplicative update stays nonnegative.

Poisson log-likelihood (up to the additive constant ``-Σ log(y_i!)``)
----------------------------------------------------------------------
``L(x) = Σ_i [ y_i log(ŷ_i) - ŷ_i ]`` for ``ŷ_i > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from petsim.reconstruction.projectors import backproject, forward_project, sensitivity_map
from petsim.reconstruction.sinogram import Sinogram

ProgressCallback = Callable[[int, int, float, float], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class MLEMHistory:
    """Per-iteration diagnostics. Iteration numbers are 1-based."""

    iterations: np.ndarray
    log_likelihood: np.ndarray
    relative_change: np.ndarray
    likelihood_note: str = (
        "Poisson log-likelihood Σ (y log ŷ − ŷ), omitting Σ log(y!); "
        "not normalized by the number of bins"
    )


def poisson_log_likelihood(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Return ``Σ (y log ŷ - ŷ)``, or ``-inf`` when ``y > 0`` and ``ŷ = 0``."""
    measured = np.asarray(y, dtype=np.float64)
    predicted = np.asarray(y_hat, dtype=np.float64)
    if np.any((measured > 0.0) & (predicted <= 0.0)):
        return float("-inf")
    total = -float(np.sum(predicted))
    positive = predicted > 0.0
    if np.any(positive):
        total += float(np.sum(measured[positive] * np.log(predicted[positive])))
    return total


def mlem_reconstruct(
    sinogram: Sinogram,
    *,
    n_iterations: int = 10,
    image_size: int = 128,
    fov: float = 1.0,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> tuple[np.ndarray, MLEMHistory, bool]:
    """Run MLEM. Returns ``(image, history, cancelled)``.

    ``cancelled`` is True when ``cancel()`` requested a stop. The image is the
    last completed iterate (or the nonnegative initial image if none finished).
    """
    n_iter = max(int(n_iterations), 1)
    n = int(image_size)
    y = np.asarray(sinogram.data, dtype=np.float64)
    thetas = np.asarray(sinogram.theta_centers, dtype=np.float64)
    s_centers = np.asarray(sinogram.s_centers, dtype=np.float64)

    if n <= 0 or y.size == 0 or thetas.size == 0 or s_centers.size == 0:
        empty = MLEMHistory(
            iterations=np.zeros(0, dtype=np.int64),
            log_likelihood=np.zeros(0, dtype=np.float64),
            relative_change=np.zeros(0, dtype=np.float64),
        )
        return np.zeros((max(n, 0), max(n, 0)), dtype=np.float64), empty, False

    sens = sensitivity_map(thetas, s_centers, image_size=n, fov=fov)
    support = sens > 0.0
    image = np.zeros((n, n), dtype=np.float64)
    image[support] = 1.0

    iterations: list[int] = []
    likelihoods: list[float] = []
    changes: list[float] = []
    cancelled = False

    for k in range(1, n_iter + 1):
        if cancel is not None and cancel():
            cancelled = True
            break
        previous = image
        y_hat = forward_project(previous, thetas, s_centers, fov=fov)
        ratio = np.zeros_like(y)
        predictable = y_hat > 0.0
        ratio[predictable] = y[predictable] / y_hat[predictable]
        correction = backproject(
            ratio, thetas, s_centers, image_size=n, fov=fov
        )
        updated = np.zeros_like(previous)
        updated[support] = previous[support] * (correction[support] / sens[support])
        # Multiplicative update on a nonnegative start stays nonnegative;
        # clip only tiny negative roundoff.
        np.maximum(updated, 0.0, out=updated)
        denom = float(np.linalg.norm(previous))
        rel = float(np.linalg.norm(updated - previous) / denom) if denom > 0.0 else 0.0
        image = updated
        # Likelihood of the updated estimate (one extra forward projection).
        y_hat_updated = forward_project(image, thetas, s_centers, fov=fov)
        ll = poisson_log_likelihood(y, y_hat_updated)
        iterations.append(k)
        likelihoods.append(ll)
        changes.append(rel)
        if progress is not None:
            progress(k, n_iter, ll, rel)

    history = MLEMHistory(
        iterations=np.asarray(iterations, dtype=np.int64),
        log_likelihood=np.asarray(likelihoods, dtype=np.float64),
        relative_change=np.asarray(changes, dtype=np.float64),
    )
    return image, history, cancelled
