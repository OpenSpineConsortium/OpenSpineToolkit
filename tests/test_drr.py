import numpy as np
import pytest

from ostk import drr, metrics, project2d
from ostk import geometry as g
from ostk.labels import lid


def _phantom(D=96):
    """Same construction as tests/test_cli.py's _phantom_label, plus a
    synthetic HU volume (bone=800 inside labeled structures, air=-1000
    elsewhere) so drr has something to integrate."""
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
    flat[ball([22, 48, 30], 9.0)] = lid("left_hip")
    flat[ball([74, 48, 30], 9.0)] = lid("right_hip")
    tilts = np.linspace(12.0, -18.0, len(metrics.LL_ENDPLATE_CHAIN))
    zs = np.linspace(80.0, 40.0, len(metrics.LL_ENDPLATE_CHAIN))
    for lv, t, z in zip(metrics.LL_ENDPLATE_CHAIN, tilts, zs):
        a = np.deg2rad(t)
        flat[body([48, 48, z], [0.0, np.sin(a), np.cos(a)])] = lid(lv)

    label = label.reshape(D, D, D)
    ct = np.full((D, D, D), -1000.0, dtype=np.float32)
    ct[label > 0] = 800.0
    return label, ct


def test_sagittal_drr_from_label_basic_shape():
    label, ct = _phantom()
    affine = np.eye(4)
    r = drr.sagittal_drr_from_label(label, ct, affine, pixel_spacing_mm=2.0)
    assert r is not None
    H, W = r["shape"]
    assert r["image"].shape == (H, W)
    assert r["pixel_spacing_mm"] == 2.0
    width_mm, height_mm = r["fov_mm"]
    assert abs(W - round(width_mm / 2.0)) <= 1
    assert abs(H - round(height_mm / 2.0)) <= 1
    # image is normalised to [0,1] and non-degenerate (bone was actually rendered)
    assert r["image"].min() >= 0.0 and r["image"].max() <= 1.0 + 1e-6
    assert r["image"].max() > 0.0


def test_sagittal_drr_from_label_respects_min_fov():
    label, ct = _phantom()
    affine = np.eye(4)
    r = drr.sagittal_drr_from_label(label, ct, affine, pixel_spacing_mm=1.0,
                                    min_fov_mm=(500.0, 700.0))
    width_mm, height_mm = r["fov_mm"]
    assert width_mm >= 500.0 - 1e-6 and height_mm >= 700.0 - 1e-6


def test_sagittal_drr_from_label_none_on_empty_volume():
    D = 32
    label = np.zeros((D, D, D), dtype=np.int16)
    ct = np.full((D, D, D), -1000.0, dtype=np.float32)
    assert drr.sagittal_drr_from_label(label, ct, np.eye(4)) is None


def test_sagittal_drr_axes_match_project2d_femoral_head_collapse():
    """The core property this module exists for: since `lr` is defined as
    unit(cR-cL), the femoral heads must project to (nearly) the same 2-D
    point under this module's own axes -- the same property project2d.py's
    landmark projection already demonstrates, and the reason xrsp's DRRs
    (using a different, approximate axis) do NOT show this collapse."""
    label, ct = _phantom()
    affine = np.eye(4)
    axis_info = drr._pi_axis_and_origin(label, affine)
    assert axis_info is not None
    ant, cranial = project2d.sagittal_axes(axis_info["lr"])
    uv_L = g.project_to_plane_2d(axis_info["cL"], axis_info["bicox"], ant, cranial)
    uv_R = g.project_to_plane_2d(axis_info["cR"], axis_info["bicox"], ant, cranial)
    assert np.linalg.norm(uv_L - uv_R) < 1e-6


def test_pixel_spacing_scales_shape_inversely():
    label, ct = _phantom()
    affine = np.eye(4)
    r1 = drr.sagittal_drr_from_label(label, ct, affine, pixel_spacing_mm=1.0)
    r2 = drr.sagittal_drr_from_label(label, ct, affine, pixel_spacing_mm=2.0)
    H1, W1 = r1["shape"]
    H2, W2 = r2["shape"]
    assert abs(H1 / 2 - H2) <= 1
    assert abs(W1 / 2 - W2) <= 1
