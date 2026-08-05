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


def _plate_with_pathology(seed=0):
    """A gently biconcave endplate carrying an osteophyte lip and a Schmorl's divot."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-18, 18, 120)
    z = 0.004 * x ** 2 + 0.05 * x + rng.normal(0, 0.15, len(x))
    z[:6] -= 9.0            # anterior spur diving away from the plate
    z[60:66] -= 4.0         # Schmorl's node divot mid-plate
    return x, z


def test_profile_model_ignores_osteophyte_and_schmorl_node():
    from ostk.vertebral_body import fit_profile_robust
    x, z = _plate_with_pathology()
    coef, w = fit_profile_robust(x, z, degree=2)
    assert abs(coef[0] - 0.004) < 0.002        # curvature recovered
    assert abs(coef[1] - 0.05) < 0.02          # slope recovered
    assert (w[:6] == 0).all()                  # spur carries zero weight
    assert (w[60:66] == 0).all()               # divot carries zero weight
    assert (w[10:55] > 0).all()                # the clean plate is fully retained


def test_profile_model_degree_two_cannot_absorb_pathology():
    """Degree matters: a higher-order model would fit the divot and defeat the point."""
    from ostk.vertebral_body import fit_profile_robust
    x, z = _plate_with_pathology()
    _, w2 = fit_profile_robust(x, z, degree=2)
    _, w8 = fit_profile_robust(x, z, degree=8)
    assert (w2[60:66] == 0).all()
    assert (w8[60:66] > 0).sum() >= (w2[60:66] > 0).sum()


def test_profile_model_needs_enough_points():
    from ostk.vertebral_body import fit_profile_robust
    assert fit_profile_robust(np.arange(3.0), np.arange(3.0), degree=2) is None
