"""Two-dimensional filtered backprojection (FBP) for parallel-beam PET.

Algorithm (educational implementation)
--------------------------------------
1. For each projection angle θ, take the 1-D radial profile p(s, θ).
2. Zero-pad and FFT along ``s``.
3. Multiply by the ramp filter ``|ω|``, optionally windowed with a Hann taper.
4. Inverse FFT → filtered projection q(s, θ).
5. Backproject: for every image point (x, y),

       f(x, y) ≈ (π / N_θ) Σ_θ  q(x cos θ + y sin θ, θ)

   with linear interpolation in ``s``.

Assumptions / limitations
-------------------------
* Parallel-beam geometry (LOR normal-form).
* No attenuation, scatter, or noise model inside FBP itself.
* A 48-crystal ring under-samples angles → streak / angular artifacts.
* Output may contain small negative values (filter Gibbs ringing); those are
  preserved in the returned array and only handled at display time.

This module uses NumPy only (no opaque external reconstructor).
"""

from __future__ import annotations

import numpy as np

from petsim.reconstruction.sinogram import Sinogram


def _ramp_filter(
    n_padded: int,
    ds: float,
    *,
    window: str = "ramp",
) -> np.ndarray:
    """Build a real-valued FFT-domain ramp (optionally Hann-windowed).

    Frequency samples follow ``numpy.fft.fftfreq`` ordering. The continuous
    ramp for parallel-beam FBP is ``H(ω) = |ω|``; with sample spacing ``ds``
    the digital frequencies are ``ξ = fftfreq(N, d=ds)`` (cycles per unit),
    and we use ``H = |2π ξ|`` in radian frequency, which is proportional to
    ``|ξ|``. Using ``|fftfreq|`` scaled by ``1/ds`` keeps relative weighting
    correct; an overall scale is absorbed in the backprojection factor.
    """
    freqs = np.fft.fftfreq(n_padded, d=ds).astype(np.float64)
    ramp = np.abs(freqs)
    # Convert cycles → a ramp proportional to |ω|; 2π keeps textbook form.
    ramp *= 2.0 * np.pi

    if window == "hann":
        # Hann taper in frequency domain, peak at DC bin index 0...N/2.
        # Apply to the magnitude shape: w(ξ) = 0.5 (1 + cos(π ξ / ξ_nyquist)).
        xi_norm = np.abs(freqs) / (0.5 / ds + 1e-15)  # 0 at DC → 1 at Nyquist
        xi_norm = np.clip(xi_norm, 0.0, 1.0)
        hann = 0.5 * (1.0 + np.cos(np.pi * xi_norm))
        ramp *= hann
    elif window not in {"ramp", "none"}:
        raise ValueError(f"Unknown filter window: {window!r}")

    return ramp


def filter_sinogram(
    sinogram: Sinogram,
    *,
    filter_name: str = "ramp",
) -> np.ndarray:
    """Return ramp-filtered projections with shape ``(n_theta, n_s)``."""
    p = np.asarray(sinogram.data, dtype=np.float64)
    n_theta, n_s = p.shape
    if n_s == 0 or n_theta == 0:
        return p.copy()

    ds = float(sinogram.s_edges[1] - sinogram.s_edges[0]) if n_s >= 1 else 1.0
    # Zero-pad to at least 2× length (next power of two) to reduce wrap-around.
    n_padded = 1 << int(np.ceil(np.log2(max(2 * n_s, 2))))
    window = "hann" if filter_name.lower() in {"hann", "hann-windowed ramp", "hanning"} else "ramp"
    h = _ramp_filter(n_padded, ds, window=window)

    filtered = np.zeros_like(p)
    for i in range(n_theta):
        row = p[i]
        padded = np.zeros(n_padded, dtype=np.float64)
        padded[:n_s] = row
        spectrum = np.fft.fft(padded)
        spectrum *= h
        q = np.real(np.fft.ifft(spectrum))
        filtered[i] = q[:n_s]
    return filtered


def backproject(
    filtered: np.ndarray,
    sinogram: Sinogram,
    *,
    image_size: int = 128,
    fov: float = 1.0,
) -> np.ndarray:
    """Pixel-driven backprojection of filtered projections onto a square grid.

    The image spans ``[-fov, fov]`` in both x and y (same units as the
    Stage 1/2 FOV). Linear interpolation is used along ``s``.
    """
    n_theta, n_s = filtered.shape
    image_size = int(image_size)
    if n_theta == 0 or n_s == 0 or image_size <= 0:
        return np.zeros((max(image_size, 0), max(image_size, 0)), dtype=np.float64)

    axis = np.linspace(-fov, fov, image_size, dtype=np.float64)
    # meshgrid with ij → yy rows, xx columns; imshow origin='lower' matches this.
    yy, xx = np.meshgrid(axis, axis, indexing="ij")

    thetas = sinogram.theta_centers
    s_coords = sinogram.s_centers
    accum = np.zeros((image_size, image_size), dtype=np.float64)

    for i, theta in enumerate(thetas):
        s_vals = xx * np.cos(theta) + yy * np.sin(theta)
        accum += np.interp(s_vals, s_coords, filtered[i], left=0.0, right=0.0)

    # Classic parallel-beam scale factor π / N_θ.
    accum *= np.pi / float(n_theta)
    return accum


def filtered_backprojection(
    sinogram: Sinogram,
    *,
    filter_name: str = "ramp",
    image_size: int = 128,
    fov: float = 1.0,
) -> np.ndarray:
    """Full FBP: filter then backproject. Returns a ``(N, N)`` float64 image.

    Negative values from filter ringing are **kept** (not clipped here).
    """
    if sinogram.data.size == 0 or not np.any(sinogram.data):
        return np.zeros((image_size, image_size), dtype=np.float64)

    filtered = filter_sinogram(sinogram, filter_name=filter_name)
    return backproject(filtered, sinogram, image_size=image_size, fov=fov)
