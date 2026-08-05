"""Tests for ostk.vertebral_body.

fit_plane_ransac had NO test when it was first written, and a shape bug in a later
vectorisation of it passed the whole 72-test suite untouched -- nothing exercised it.
"""
import numpy as np

from ostk.vertebral_body import fit_plane_ransac


def _plane_with_outliers(n=400, frac_out=0.2, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(-20, 20, n), rng.uniform(-20, 20, n), rng.normal(0, noise, n)]
    k = int(frac_out * n)
    P[:k, 2] += 25.0                       # a gross, coherent outlier slab
    return P, k


def test_ransac_recovers_plane_despite_gross_outliers():
    P, k = _plane_with_outliers()
    out = fit_plane_ransac(P, thresh_mm=1.5)
    assert out is not None
    c, nrm, inl = out
    assert abs(abs(float(nrm[2])) - 1.0) < 1e-2        # z-normal
    assert abs(float(c[2])) < 1.0                      # centroid on z=0
    assert inl.sum() >= 0.9 * (len(P) - k)             # keeps the true inliers
    assert not inl[:k].any()                           # rejects every outlier


def test_ransac_returns_none_on_degenerate_input():
    assert fit_plane_ransac(np.zeros((2, 3))) is None


def test_ransac_is_deterministic():
    P, _ = _plane_with_outliers()
    a = fit_plane_ransac(P, thresh_mm=1.5)
    b = fit_plane_ransac(P, thresh_mm=1.5)
    assert np.allclose(a[1], b[1]) and np.allclose(a[0], b[0])
