from __future__ import annotations

import numpy as np


class WindField:
    """Time-varying, spatially non-uniform wind field.

    N fixed kernel centers are sampled at reset within the map. Each kernel carries a 2D
    wind vector that evolves as an independent Ornstein-Uhlenbeck process (mean-reverting
    to zero). Wind at any point (x, y) is the RBF-weighted average of kernel vectors:

        w_k(x, y) = exp(-||(x,y) - c_k||^2 / (2 sigma^2))
        wind(x, y, t) = sum_k w_k v_k / sum_k w_k
    """

    __slots__ = (
        "map_range", "n_kernels", "sigma", "ou_theta", "ou_sigma", "max_speed",
        "_rng", "centers", "kernel_wind",
    )

    def __init__(self, map_range: float, n_kernels: int, sigma: float,
                 ou_theta: float, ou_sigma: float, max_speed: float,
                 rng: np.random.Generator):
        self.map_range = float(map_range)
        self.n_kernels = int(n_kernels)
        self.sigma = max(float(sigma), 1e-6)
        self.ou_theta = max(float(ou_theta), 1e-9)
        self.ou_sigma = max(float(ou_sigma), 0.0)
        self.max_speed = max(float(max_speed), 0.0)
        self._rng = rng

        margin = 0.10 * self.map_range
        self.centers = rng.uniform(margin, self.map_range - margin,
                                   size=(self.n_kernels, 2)).astype(np.float64)

        # Initialize at OU stationary distribution N(0, sigma^2 / (2 theta)).
        sigma_stat = self.ou_sigma / np.sqrt(2.0 * self.ou_theta)
        self.kernel_wind = rng.normal(0.0, sigma_stat,
                                      size=(self.n_kernels, 2)).astype(np.float64)
        np.clip(self.kernel_wind, -self.max_speed, self.max_speed, out=self.kernel_wind)

    def advance(self, dt: float) -> None:
        """Time-evolve each kernel's wind by dt using analytical OU update."""
        if dt <= 0.0:
            return
        theta = self.ou_theta
        sigma = self.ou_sigma
        exp_dt = float(np.exp(-theta * dt))
        var_term = (1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta)
        noise_scale = sigma * np.sqrt(max(var_term, 0.0))
        noise = self._rng.standard_normal((self.n_kernels, 2)) * noise_scale
        self.kernel_wind = self.kernel_wind * exp_dt + noise
        np.clip(self.kernel_wind, -self.max_speed, self.max_speed, out=self.kernel_wind)

    def get(self, xy: np.ndarray) -> np.ndarray:
        """Return wind vector (2,) at position xy."""
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        diff = self.centers - xy
        dist_sq = np.sum(diff * diff, axis=1)
        m = float(np.min(dist_sq))
        w = np.exp(-(dist_sq - m) / (2.0 * self.sigma * self.sigma))
        w_sum = float(np.sum(w))
        if w_sum < 1e-12:
            return np.zeros(2, dtype=np.float64)
        return (self.kernel_wind * w[:, None]).sum(axis=0) / w_sum
