import math

import numpy as np
import pytest

from ostk import geometry as g
from ostk import metrics
from ostk import project2d
from ostk.labels import lid


def _angle_2d(v1, v2):
    """Angle (deg, unsigned) between two 2D vectors -- scale-invariant, so it
    works whether or not project_to_plane_2d's output was renormalised."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return abs(math.degrees(math.atan2(cross, dot)))


# --- geometry.project_to_plane_2d: pure primitive --------------------------

def test_project_to_plane_2d_matches_manual_dot_products():
    origin = np.array([1.0, 2.0, 3.0])
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    p = np.array([5.0, 9.0, 100.0])          # z is out-of-plane, must be dropped
    uv = g.project_to_plane_2d(p, origin, u, v)
    assert np.allclose(uv, [(p - origin) @ u, (p - origin) @ v])


def test_project_to_plane_2d_batch_matches_pointwise():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3))
    origin, u, v = np.zeros(3), np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    batch = g.project_to_plane_2d(pts, origin, u, v)
    single = np.array([g.project_to_plane_2d(p, origin, u, v) for p in pts])
    assert np.allclose(batch, single)


def test_project_to_plane_2d_angle_matches_3d_signed_angle_in_plane():
    """The core equivalence claim: an angle computed from explicit 2D
    coordinates must equal geometry.signed_angle_in_plane's 3D computation on
    the same two vectors, confined to the same plane."""
    lr = np.array([1.0, 0.0, 0.0])            # plane normal (L-R axis)
    ant, cranial = project2d.sagittal_axes(lr)

    def endplate_normal(tilt_deg):
        a = np.deg2rad(tilt_deg)
        return np.array([0.0, np.sin(a), np.cos(a)])

    n_top, n_bot = endplate_normal(15.0), endplate_normal(-25.0)
    expected = g.signed_angle_in_plane(n_top, n_bot, lr)

    top2d = g.project_to_plane_2d(n_top, np.zeros(3), ant, cranial)
    bot2d = g.project_to_plane_2d(n_bot, np.zeros(3), ant, cranial)
    got = math.copysign(_angle_2d(top2d, bot2d), expected)
    assert abs(got - expected) < 1e-6


# --- pi_landmarks_2d: pure point-cloud function -----------------------------

def _sphere_points(center, radius, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.asarray(center) + radius * d


def _plane_points(midpoint, normal, half=40.0, step=4.0):
    n = g.unit(normal)
    u1 = g.unit(g.project_out(np.array([1.0, 0, 0]), n))
    u2 = g.unit(np.cross(n, u1))
    ts = np.arange(-half, half + step, step)
    a, b = np.meshgrid(ts, ts)
    grid = a.ravel()[:, None] * u1 + b.ravel()[:, None] * u2
    return np.asarray(midpoint) + grid


def test_pi_landmarks_2d_reproduces_pi_from_metrics():
    """Same phantom as test_metrics.test_pelvic_incidence_phantom_known_angles
    (analytic PI=60, SS=40). Recomputing PI/SS from the 2D landmarks must
    match metrics.pelvic_incidence's (unmodified) 3D result."""
    cL, cR = np.array([-80.0, 0, 0]), np.array([80.0, 0, 0])
    PT, SS = np.deg2rad(20.0), np.deg2rad(40.0)
    P = 150.0 * np.array([0.0, np.sin(PT), np.cos(PT)])
    n = np.array([0.0, -np.sin(SS), np.cos(SS)])

    ep = _plane_points(P, n)
    fhl = _sphere_points(cL, 25.0, seed=1)
    fhr = _sphere_points(cR, 25.0, seed=2)

    r3d = metrics.pelvic_incidence(ep, fhl, fhr)
    r2d = project2d.pi_landmarks_2d(ep, fhl, fhr)

    lm = r2d["landmarks_2d_mm"]
    fhl2, fhr2 = np.array(lm["femhead_left"]), np.array(lm["femhead_right"])
    ep2 = np.array(lm["endplate_midpoint"])
    n2 = np.array(r2d["endplate_normal_2d"])

    radius2 = ep2 - 0.5 * (fhl2 + fhr2)       # bicox -> endplate midpoint, in 2D
    vertical2 = np.array([0.0, 1.0])          # cranial axis == 2D "up" by construction

    PI2 = _angle_2d(n2, radius2)
    SS2 = _angle_2d(n2, vertical2)

    assert abs(PI2 - r3d["PI"]) < 0.5
    assert abs(SS2 - r3d["SS"]) < 0.5
    # cL/cR differ only along the L-R axis itself (the projection axis), so an
    # orthographic lateral-view projection correctly collapses them to the
    # same 2D point -- this is the real "femoral heads overlap on a lateral
    # film" effect, not a projection bug.
    assert np.linalg.norm(fhl2 - fhr2) < 1e-6


def test_pi_landmarks_2d_bicoxofemoral_is_origin():
    cL, cR = np.array([-70.0, 0, 0]), np.array([70.0, 0, 0])
    P, n = np.array([0.0, 40.0, 130.0]), np.array([0.0, -0.5, 0.8])
    r2d = project2d.pi_landmarks_2d(_plane_points(P, n), _sphere_points(cL, 22.0, seed=5),
                                    _sphere_points(cR, 22.0, seed=6))
    assert np.allclose(r2d["landmarks_2d_mm"]["bicoxofemoral"], [0.0, 0.0], atol=1e-9)


# --- ll_landmarks_2d: pure point-cloud function -----------------------------

def test_ll_landmarks_2d_reproduces_ll_from_metrics():
    lr = np.array([1.0, 0.0, 0.0])
    tilts = np.linspace(15.0, -25.0, len(metrics.LL_ENDPLATE_CHAIN))
    normals = {lv: np.array([0.0, np.sin(np.deg2rad(t)), np.cos(np.deg2rad(t))])
              for lv, t in zip(metrics.LL_ENDPLATE_CHAIN, tilts)}
    centroids = {lv: np.array([0.0, 0.0, -20.0 * i])
                for i, lv in enumerate(metrics.LL_ENDPLATE_CHAIN)}

    r3d = metrics.lumbar_lordosis(normals, lr)
    r2d = project2d.ll_landmarks_2d(normals, centroids, lr)

    top, bot = r3d["levels"][0], r3d["levels"][-1]
    n_top2 = np.array(r2d["endplate_normals_2d"][top])
    n_bot2 = np.array(r2d["endplate_normals_2d"][bot])
    assert abs(_angle_2d(n_top2, n_bot2) - r3d["LL"]) < 1e-6


def test_ll_landmarks_2d_needs_two_levels():
    lr = np.array([1.0, 0.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    r = project2d.ll_landmarks_2d({"L1": n}, {"L1": np.zeros(3)}, lr)
    assert r["landmarks_2d_mm"] == {} and r["levels"] == ["L1"]


# --- *_2d_from_label wrappers: label-volume smoke test ----------------------

def _phantom_label(D=96):
    ijk = np.argwhere(np.ones((D, D, D), dtype=bool)).astype(float)
    label = np.zeros((D, D, D), dtype=np.int16)
    flat = label.reshape(-1)

    def ball(c, r):
        return np.linalg.norm(ijk - np.asarray(c), axis=1) <= r

    def body(c, normal, radius=18.0, half=7.0):
        n = g.unit(normal)
        d = (ijk - np.asarray(c)) @ n
        inplane = np.linalg.norm((ijk - np.asarray(c)) - np.outer(d, n), axis=1)
        return (np.abs(d) <= half) & (inplane <= radius)

    flat[ball([22, 48, 22], 13.0)] = lid("femur_left")
    flat[ball([74, 48, 22], 13.0)] = lid("femur_right")
    tilts = np.linspace(12.0, -18.0, len(metrics.LL_ENDPLATE_CHAIN))
    zs = np.linspace(80.0, 40.0, len(metrics.LL_ENDPLATE_CHAIN))
    for lv, t, z in zip(metrics.LL_ENDPLATE_CHAIN, tilts, zs):
        a = np.deg2rad(t)
        flat[body([48, 48, z], [0.0, np.sin(a), np.cos(a)])] = lid(lv)
    return label


def test_pelvic_incidence_2d_from_label_matches_pi_measurement():
    lab = _phantom_label()
    affine = np.eye(4)
    m = metrics.pelvic_incidence_from_label(lab, affine, case_id="p1")
    r2d = project2d.pelvic_incidence_2d_from_label(lab, affine)

    assert m.value is not None and r2d is not None
    lm = r2d["landmarks_2d_mm"]
    fhl2, fhr2 = np.array(lm["femhead_left"]), np.array(lm["femhead_right"])
    ep2, n2 = np.array(lm["endplate_midpoint"]), np.array(r2d["endplate_normal_2d"])
    radius2 = ep2 - 0.5 * (fhl2 + fhr2)
    PI2 = _angle_2d(n2, radius2)
    assert abs(PI2 - m.value) < 1.0


def test_lumbar_lordosis_2d_from_label_matches_ll_measurement():
    lab = _phantom_label()
    affine = np.eye(4)
    m = metrics.lumbar_lordosis_from_label(lab, affine, case_id="p1")
    r2d = project2d.lumbar_lordosis_2d_from_label(lab, affine)

    assert m.value is not None and r2d is not None
    top, bot = r2d["levels"][0], r2d["levels"][-1]
    n_top2 = np.array(r2d["endplate_normals_2d"][top])
    n_bot2 = np.array(r2d["endplate_normals_2d"][bot])
    assert abs(_angle_2d(n_top2, n_bot2) - m.value) < 1.0


def test_2d_from_label_none_on_empty_volume():
    lab = np.zeros((32, 32, 32), dtype=np.int16)
    affine = np.eye(4)
    assert project2d.pelvic_incidence_2d_from_label(lab, affine) is None
    assert project2d.lumbar_lordosis_2d_from_label(lab, affine) is None


# --- backward-compatibility guard: Measurement's shape is untouched --------

def test_measurement_to_dict_keys_unchanged():
    """This whole feature was scoped as opt-in specifically so Measurement's
    dict shape (and therefore CLI CSV/JSONL output) doesn't change. Guard
    that promise explicitly."""
    from ostk.record import Measurement
    m = Measurement(case_id="x", parameter="p", value=1.0)
    assert set(m.to_dict().keys()) == {
        "case_id", "parameter", "value", "units", "landmarks_world_mm",
        "fit_residuals", "qc_flags", "method_version", "supine_ct",
    }
