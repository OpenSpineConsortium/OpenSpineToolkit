"""ostk.project2d -- opt-in orthographic 2D sagittal-plane projection of the
same landmarks PI/SS/PT/LL are already computed from.

PI/SS/PT/LL (ostk.metrics) are already angles between 3D vectors that have
been projected onto the patient sagittal plane (see geometry.cobb_angle /
signed_angle_in_plane's docstrings) -- so nothing in this module changes any
angle value. It adds an explicit 2D (anterior, cranial) millimetre coordinate
system for the same landmarks, for rendering or comparison against a
lateral-radiograph-style view.

Projection is orthographic (parallel rays) along the patient's data-derived
bicoxofemoral (L-R) axis -- i.e. it assumes ideal lateral positioning, the
same assumption already implicit in the 3D angle pipeline. It does NOT model
an X-ray's point-source cone-beam magnification/parallax, and it does not
simulate positioning/rotation error -- the projection axis is always the true
CT-derived L-R axis, never a perturbed one.

Mirrors ostk.metrics' two-tier layout: a pure point-cloud function
(``pi_landmarks_2d`` / ``ll_landmarks_2d``) plus a ``*_2d_from_label`` wrapper
that extracts point clouds from a label volume, the same way metrics.py's
``*_from_label`` functions do. Nothing elsewhere in ostk imports this module --
it is purely additive/opt-in: existing Measurement objects, the CLI, and
CSV/JSONL output are unaffected whether or not this module is ever used.

Note on duplication: ``pelvic_incidence_2d_from_label`` mirrors (rather than
imports) ``metrics._pi_from_label_core``'s point-cloud extraction, because
that private function returns only final angles/landmarks, not the raw
endplate normal or L-R axis a 2D projection needs. If that extraction logic
changes, this should be updated to match. ``lumbar_lordosis_2d_from_label``
has no such gap -- it reuses ``metrics._lr_axis_from_label`` and
``metrics._endplate_normal_from_label`` directly (the same cross-module reuse
of metrics.py's private per-case helpers that ostk.cobb already does).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import (WORLD_SUPERIOR, fit_plane_tls, fit_sphere, project_out,
                       project_to_plane_2d, unit)
from .metrics import LL_ENDPLATE_CHAIN
from .spine import anterior_axis


def sagittal_axes(lr_axis, sup_axis=WORLD_SUPERIOR):
    """Orthonormal in-plane (anterior, cranial) axes for the sagittal plane
    whose normal is `lr_axis`. `anterior` uses the same convention as
    ostk.spine.anterior_axis (shared with the coronal Cobb pipeline);
    `cranial` is `sup_axis` projected into the plane."""
    return anterior_axis(sup_axis, lr_axis), unit(project_out(sup_axis, lr_axis))


# ---------------------------------------------------------------------------
# Pelvic incidence landmarks
# ---------------------------------------------------------------------------

def pi_landmarks_2d(endplate_points, femhead_left_points, femhead_right_points,
                    sup_axis=WORLD_SUPERIOR) -> Dict:
    """Orthographic 2D sagittal-plane projection of the PI landmarks, from the
    same three world-mm point clouds metrics.pelvic_incidence takes. Returns
    2D (anterior, cranial) mm coordinates for the femoral heads, their
    midpoint (the 2D origin), and the S1/sacrum endplate midpoint + normal --
    everything metrics.pelvic_incidence's PI/SS/PT are computed from, just in
    an explicit 2D basis instead of a 3D vector with the L-R component zeroed."""
    m, n, ep_rms = fit_plane_tls(endplate_points)
    cL, rL, eL = fit_sphere(femhead_left_points)
    cR, rR, eR = fit_sphere(femhead_right_points)
    bicox = 0.5 * (cL + cR)
    lr = unit(cR - cL)
    ant, cranial = sagittal_axes(lr, sup_axis)

    return {
        "origin_world_mm": bicox.tolist(),
        "axes": {"anterior": ant.tolist(), "cranial": cranial.tolist()},
        "landmarks_2d_mm": {
            "femhead_left": project_to_plane_2d(cL, bicox, ant, cranial).tolist(),
            "femhead_right": project_to_plane_2d(cR, bicox, ant, cranial).tolist(),
            "bicoxofemoral": project_to_plane_2d(bicox, bicox, ant, cranial).tolist(),
            "endplate_midpoint": project_to_plane_2d(m, bicox, ant, cranial).tolist(),
        },
        "endplate_normal_2d": project_to_plane_2d(n, np.zeros(3), ant, cranial).tolist(),
        "fit_residuals": {
            "s1_endplate_rms": ep_rms,
            "femhead_left_rms": eL, "femhead_right_rms": eR,
            "femhead_left_radius": rL, "femhead_right_radius": rR,
        },
    }


def pelvic_incidence_2d_from_label(label, affine, *, sup_axis=WORLD_SUPERIOR,
                                   endplate_frac: float = 0.15,
                                   head_frac: float = 0.35,
                                   min_voxels: int = 50) -> Optional[Dict]:
    """Label-volume wrapper for pi_landmarks_2d. Extracts the same three
    point clouds ostk.metrics._pi_from_label_core does (S1/sacrum endplate
    slab, femoral head slabs) and mirrors its QC gate (returns None if any is
    too small) -- see the module docstring for why this duplicates that
    extraction rather than importing it."""
    from .labels import lid
    from .masks import (binary_mask, endplate_points, largest_component,
                        mask_world, surface_slab)

    s1 = binary_mask(label, lid("S1"))
    src = s1 if s1.any() else binary_mask(label, lid("sacrum"))
    ep = endplate_points(largest_component(src), affine, sup_axis, "superior",
                         endplate_frac)
    fl = mask_world(largest_component(binary_mask(label, lid("femur_left"))), affine)
    fr = mask_world(largest_component(binary_mask(label, lid("femur_right"))), affine)
    fhl = surface_slab(fl, sup_axis, "superior", head_frac)
    fhr = surface_slab(fr, sup_axis, "superior", head_frac)
    if min(len(ep), len(fhl), len(fhr)) < min_voxels:
        return None
    return pi_landmarks_2d(ep, fhl, fhr, sup_axis)


# ---------------------------------------------------------------------------
# Lumbar lordosis landmarks
# ---------------------------------------------------------------------------

def ll_landmarks_2d(endplate_normals: Dict[str, np.ndarray],
                    endplate_centroids: Dict[str, np.ndarray], lr_axis,
                    sup_axis=WORLD_SUPERIOR) -> Dict:
    """Orthographic 2D sagittal-plane projection of the per-level endplate
    landmarks metrics.lumbar_lordosis's LL/segment Cobb angles are computed
    from. `endplate_normals`/`endplate_centroids` share keys (a cranial->caudal
    chain such as metrics.LL_ENDPLATE_CHAIN); the first present level's
    centroid is the 2D origin."""
    present = [lv for lv in LL_ENDPLATE_CHAIN if lv in endplate_normals]
    if len(present) < 2:
        return {"landmarks_2d_mm": {}, "endplate_normals_2d": {}, "levels": present}
    ant, cranial = sagittal_axes(lr_axis, sup_axis)
    origin = endplate_centroids[present[0]]
    landmarks_2d = {lv: project_to_plane_2d(endplate_centroids[lv], origin,
                                            ant, cranial).tolist()
                    for lv in present}
    normals_2d = {lv: project_to_plane_2d(endplate_normals[lv], np.zeros(3),
                                          ant, cranial).tolist()
                  for lv in present}
    return {
        "origin_world_mm": np.asarray(origin).tolist(),
        "axes": {"anterior": ant.tolist(), "cranial": cranial.tolist()},
        "landmarks_2d_mm": landmarks_2d, "endplate_normals_2d": normals_2d,
        "levels": present,
    }


def lumbar_lordosis_2d_from_label(label, affine, *, sup_axis=WORLD_SUPERIOR,
                                  endplate_frac: float = 0.15,
                                  head_frac: float = 0.35,
                                  min_voxels: int = 30) -> Optional[Dict]:
    """Label-volume wrapper for ll_landmarks_2d. Reuses
    metrics._lr_axis_from_label and metrics._endplate_normal_from_label (the
    same per-case extraction metrics.lumbar_lordosis_from_label uses) so the
    L-R axis and each endplate fit are identical to the 3D LL pipeline; only
    the final projection is new. Returns None if fewer than two endplates are
    available."""
    from .metrics import _endplate_normal_from_label, _lr_axis_from_label

    lr, lr_ok = _lr_axis_from_label(label, affine, sup_axis, head_frac, min_voxels)
    if not lr_ok:
        return None
    normals: Dict[str, np.ndarray] = {}
    centroids: Dict[str, np.ndarray] = {}
    for lv in LL_ENDPLATE_CHAIN:
        n, c, rms, k = _endplate_normal_from_label(
            label, affine, lv, "superior", sup_axis, endplate_frac, min_voxels)
        if n is not None:
            normals[lv], centroids[lv] = n, c
    if len(normals) < 2:
        return None
    return ll_landmarks_2d(normals, centroids, lr, sup_axis)
