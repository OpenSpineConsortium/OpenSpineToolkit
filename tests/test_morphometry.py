"""The test that matters here is INVARIANCE UNDER TILT.

The bias this module exists to remove is that a dimension taken along the scanner's axes
is the true dimension divided by the cosine of however far the bone is tilted. That is
invisible in any single number and only shows up when the same bone is measured twice at
two angles. So the phantom below is built once, measured, rotated, and measured again:
the height must not move.

An extent-based estimator fails this by construction, and the assertion at the end of
`test_tilt_invariance` shows by how much, so a regression cannot quietly reintroduce it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ostk.geometry import rotation_matrix
from ostk.morphometry import body_heights, vertebra_frame

# A vertebra 24 mm tall at the posterior wall and 27 mm at the anterior, 35 mm deep:
# roughly an L3 with the normal slight anterior wedge.
VBHA, VBHP, DEPTH = 27.0, 24.0, 35.0


def _corners(tilt_deg: float = 0.0, axis=(1.0, 0.0, 0.0)):
    """(superior, inferior) corner pairs for the phantom, optionally tilted.

    Anterior is +y, superior is +z. The anterior wall is taller, so the superior plate
    rises as it runs forward, which is what an anteriorly wedged body looks like.
    """
    sup = np.array([[0.0, DEPTH / 2, VBHA / 2], [0.0, -DEPTH / 2, VBHP / 2]])
    inf = np.array([[0.0, DEPTH / 2, -VBHA / 2], [0.0, -DEPTH / 2, -VBHP / 2]])
    if tilt_deg:
        R = rotation_matrix(np.asarray(axis, float), np.deg2rad(tilt_deg))
        sup, inf = sup @ R.T, inf @ R.T
    return (sup[0], sup[1]), (inf[0], inf[1])


def test_heights_match_the_phantom():
    sup, inf = _corners()
    h = body_heights(sup, inf)
    assert h["VBHa"] == pytest.approx(VBHA, abs=1e-9)
    assert h["VBHp"] == pytest.approx(VBHP, abs=1e-9)


@pytest.mark.parametrize("tilt", [0, 5, 10, 20, 30, -25])
def test_tilt_invariance(tilt):
    """The measured height must not depend on how the patient lay in the scanner."""
    sup, inf = _corners(tilt)
    h = body_heights(sup, inf)
    assert h["VBHa"] == pytest.approx(VBHA, abs=1e-6)
    assert h["VBHp"] == pytest.approx(VBHP, abs=1e-6)

    # and the estimator this replaces does NOT survive the same rotation: a straight-up
    # extent in the scanner's frame grows as the bone tilts away from it
    pts = np.array([sup[0], sup[1], inf[0], inf[1]])
    world_extent = float(pts[:, 2].max() - pts[:, 2].min())
    if abs(tilt) >= 10:
        assert world_extent > VBHA + 0.5, (
            f"tilt {tilt} deg should inflate a world-axis extent, got {world_extent:.2f}")


def test_frame_is_orthonormal_and_laterally_oriented():
    """Orthonormal, and the lateral axis points the way the caller's does.

    HANDEDNESS IS DELIBERATELY NOT ASSERTED. The lateral axis is flipped when needed so
    that "left" means the same side at every level, which is what the per-side pedicle
    names depend on; whether the resulting triple is right- or left-handed then follows
    from which way left-right runs in the volume, and is a property of the data. Nothing
    in this module reads a sign off the frame -- every dimension is a span or a distance,
    both of which are sign-free -- so enforcing handedness would trade a meaningful
    guarantee for a meaningless one.
    """
    sup, inf = _corners(17.0)
    f = vertebra_frame(sup, inf, lr=(1.0, 0.0, 0.0))
    assert f is not None
    ap, s, lat = f["ap"], f["sup"], f["lat"]
    for v in (ap, s, lat):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)
    assert ap @ s == pytest.approx(0.0, abs=1e-9)
    assert ap @ lat == pytest.approx(0.0, abs=1e-9)
    assert s @ lat == pytest.approx(0.0, abs=1e-9)
    assert lat @ np.array([1.0, 0.0, 0.0]) > 0

    # and flipping the caller's convention flips the axis, nothing else
    g = vertebra_frame(sup, inf, lr=(-1.0, 0.0, 0.0))
    assert g["lat"] == pytest.approx(-lat, abs=1e-9)
    assert g["ap"] == pytest.approx(ap, abs=1e-9)
    assert g["sup"] == pytest.approx(s, abs=1e-9)


def test_frame_follows_the_bone_not_the_world():
    """Tilt the phantom and the frame's superior axis must tilt with it."""
    f0 = vertebra_frame(*_corners(0.0))
    f30 = vertebra_frame(*_corners(30.0))
    assert f0 is not None and f30 is not None
    ang = np.degrees(np.arccos(np.clip(f0["sup"] @ f30["sup"], -1, 1)))
    assert ang == pytest.approx(30.0, abs=1.0)


def test_degenerate_plates_return_none():
    """Two plates that give no anteroposterior direction must drop the level."""
    pt = np.zeros(3)
    assert vertebra_frame((pt, pt), (pt, pt)) is None
