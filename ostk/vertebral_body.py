"""Vertebral BODY isolation and endplate geometry.

Why this module exists
----------------------
Everything that goes wrong when you fit an endplate from a whole-vertebra mask goes
wrong for one reason: the posterior elements are still in the data. The pedicles,
lamina and spinous process sit behind the body, at endplate height, and any fit that
sees them is pulled backwards. Working around that downstream produces exactly the
symptoms we measured -- a posterior corner landing inside the spinal canal, an endplate
line crossing into the next vertebra's arch, and a `drop_post` fraction that cuts at a
different anatomic place on every level because it is a fraction of an extent that
includes the very structures it is meant to exclude.

So isolate the body FIRST. Then the endplate is unambiguous and the downstream
parameters (`drop_post`, rim bands, canal-gap searches, snap limits) are not needed at
all -- there is nothing left to exclude.

Method and prior art
--------------------
1. BODY / POSTERIOR-ELEMENT PARTITION -- the SPINAL CANAL, with watershed as a fallback.

   The canal is a topological HOLE: in an axial section the body, pedicles and lamina
   enclose it. Fill the holes slice by slice and the filled-minus-original difference IS
   the canal, with no threshold to choose. Its anterior boundary is the body's posterior
   wall -- the "posterior vertebral body line" that is read directly off a lateral film.

   A distance-transform watershed was tried first and is kept only as a fallback, because
   it assumes the body/arch junction is a THIN neck. That holds in the thoracic spine and
   fails in the lumbar, where pedicles are 8-18 mm across: measured on case 0004 it left
   L2, L3 and L4 completely unpartitioned (100% of the mask returned as "body"), and the
   endplate spans blew out to 70-92 mm as the posterior corner ran to the spinous process.
   The canal criterion has no such level dependence -- every vertebra encloses a canal.

   Watershed reference, still used when the canal cannot be recovered (an incomplete arch,
   e.g. spondylolysis, leaves no enclosed hole to fill):
   The pedicle isthmus is a thin neck joining two thick regions, which is the canonical
   watershed cut. Markers are the connected components of the mask eroded in distance
   terms (EDT >= `core_mm`), so the body core and the posterior-element core each seed
   one basin, and the flood meets at the isthmus.
     Yao J, O'Connor SD, Summers RM. "Automated spinal column extraction and
       partitioning." IEEE ISBI 2006, 390-393.
     Naegel B. "Using mathematical morphology for the anatomical labeling of vertebrae
       from 3D CT-scan images." Comput Med Imaging Graph 2007;31(3):141-156.
   The body is identified as the basin containing the global distance-transform maximum
   -- the vertebral body is the thickest part of a vertebra, which holds whether or not
   the posterior elements are complete, and needs no anterior axis to be known first.

2. BODY COORDINATE SYSTEM.
     Mastmeyer A, Engelke K, Fuchs C, Kalender WA. "A hierarchical 3D segmentation
       method and the definition of vertebral body coordinate systems for QCT of the
       lumbar spine." Med Image Anal 2006;10(4):560-577.
   Mastmeyer isolate the body for the same reason we do: in QCT the posterior arch
   corrupts the measurement. We keep the patient axes rather than the body's own PCA
   axes, because every angle in ostk is defined against the patient frame and switching
   would silently redefine SS/LL.

3. ENDPLATE PLANE -- RANSAC.
     Fischler MA, Bolles RC. "Random sample consensus: a paradigm for model fitting
       with applications to image analysis and automated cartography."
       Commun ACM 1981;24(6):381-395.
   RANSAC is the standard robust estimator for exactly this failure mode: a majority of
   inliers (the endplate) plus a minority of gross outliers (an anterior osteophyte lip,
   a Schmorl's node, a segmentation spur). It replaces iterative trimming, which has no
   breakdown guarantee and which we measured failing -- residuals of 7-12 mm against
   0.5-2.5 mm on sound levels.

4. CORNERS -- the A-P extremes of the endplate surface.
   Reported as the chord through the two corners, not the RANSAC plane itself, because
   the endplate is CONCAVE and the clinical line bridges that concavity: it is drawn
   corner to corner, the same construction a Cobb angle uses.
     Vrtovec T, Pernus F, Likar B. "A review of methods for quantitative evaluation of
       spinal curvature." Eur Spine J 2009;18(5):593-607.
     Stern D, Likar B, Pernus F, Vrtovec T. "Parametric modelling and segmentation of
       vertebral bodies in 3D CT and MR spine images." Phys Med Biol 2011;56(23):7505-22.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .geometry import WORLD_SUPERIOR, unit
from .spine import anterior_axis


def voxel_spacing(affine) -> np.ndarray:
    """Physical voxel size (mm) along each array axis."""
    return np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)


def _sup_array_axis(affine, sup_axis) -> int:
    """Array axis most closely aligned with the patient's superior direction."""
    M = np.asarray(affine, float)[:3, :3]
    return int(np.argmax(np.abs(unit(np.asarray(sup_axis, float)) @ M / np.linalg.norm(M, axis=0))))


def canal_mask(mask, affine, *, sup_axis=WORLD_SUPERIOR, min_vox: int = 20
               ) -> Optional[np.ndarray]:
    """The spinal canal enclosed by this vertebra, as a topological hole.

    Filled per axial section rather than in 3-D: in 3-D the canal is an open tube, so it
    is not a hole at all and nothing would be filled. Returns None when the arch does not
    close the ring in enough sections to be believable -- an incomplete or lytic arch --
    rather than returning a sliver that would put the wall in the wrong place.
    """
    m = np.asarray(mask, bool)
    ax = _sup_array_axis(affine, sup_axis)
    filled = np.zeros_like(m)
    mv = np.moveaxis(m, ax, 0)
    fv = np.moveaxis(filled, ax, 0)
    for k in range(mv.shape[0]):
        if mv[k].any():
            fv[k] = ndimage.binary_fill_holes(mv[k])
    canal = np.moveaxis(fv, 0, ax) & ~m
    if canal.sum() < min_vox:
        return None
    lab, n = ndimage.label(canal)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def body_mask_by_canal(mask, affine, *, sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                       margin_mm: float = 0.0, canal_from=None) -> Optional[np.ndarray]:
    """BODY = the part of the mask anterior to the canal, cut SECTION BY SECTION.

    The cut follows the posterior vertebral body line, which is CONCAVE -- it bows
    anteriorly at mid-body and runs further posterior at the endplate rims. A single
    plane at the canal's globally anterior-most point therefore cuts at the mid-body
    concavity and shaves the posterior rim off BOTH endplates, which is visible on an
    overlay as a superior corner that does not reach far enough back. So the wall is
    taken per axial section instead, and sections with no canal (above and below its
    extent) inherit the nearest section that has one.

    `canal_from` supplies a LARGER mask to find the canal in, while the cut is still
    applied to `mask`. Needed for S1, which in the v4 scheme is carved off the top of the
    sacrum: that carve can slice the arch so the ring never closes in any one section,
    and S1 alone then yields no canal at all. Measured on two cases -- 0001 S1: 0 closed
    sections, 0 hole voxels; 0004 S1: 6 and 4693. The sacral canal does not stop at an
    artificial label boundary, so pass S1 | sacrum and the ring closes (0001 sacrum: 47
    closed sections). Without this S1 silently took the watershed fallback and the
    endplate line ran across the whole sacrum into the posterior elements.

    Unlike a thickness or fraction rule this is the same anatomic landmark at every
    level, in every patient.
    """
    canal = canal_mask(mask if canal_from is None else canal_from,
                       affine, sup_axis=sup_axis)
    if canal is None:
        return None
    ap = anterior_axis(unit(sup_axis), lr)
    aff = np.asarray(affine, float)
    ax = _sup_array_axis(affine, sup_axis)
    m = np.asarray(mask, bool)

    ci = np.array(np.nonzero(canal)).T
    cw = (np.c_[ci, np.ones(len(ci))] @ aff.T)[:, :3]
    ap_c, k_c = cw @ ap, ci[:, ax]

    n_sl = m.shape[ax]
    # -inf, NOT nan: np.maximum propagates nan, so a nan-initialised accumulator stays
    # nan for every section, `have` comes back empty, and this function silently returns
    # None for EVERY vertebra -- handing all of them to the watershed fallback, which is
    # exactly the level-dependent method this replaces. It fails loudly nowhere and
    # quietly everywhere; the only visible symptom was endplate lines running through the
    # whole sacrum and L4/L5 spans of 57 mm.
    wall = np.full(n_sl, -np.inf)
    np.maximum.at(wall, k_c, ap_c)                    # anterior canal edge per section
    have = np.nonzero(np.isfinite(wall))[0]
    if not len(have):
        return None
    # sections outside the canal's extent inherit the nearest section that has one
    idx = np.clip(np.searchsorted(have, np.arange(n_sl)), 0, len(have) - 1)
    prev = np.clip(idx - 1, 0, len(have) - 1)
    pick = np.where(np.abs(have[idx] - np.arange(n_sl))
                    <= np.abs(have[prev] - np.arange(n_sl)), have[idx], have[prev])
    wall = wall[pick] + float(margin_mm)

    mi = np.array(np.nonzero(m)).T
    mw = (np.c_[mi, np.ones(len(mi))] @ aff.T)[:, :3]
    sel = (mw @ ap) >= wall[mi[:, ax]]
    if sel.sum() < 30:
        return None
    out = np.zeros_like(m)
    ii = mi[sel]
    out[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return out


def body_mask_watershed(mask, affine, *, core_mm: float = 4.0, min_core_vox: int = 30
                        ) -> Optional[np.ndarray]:
    """Split a whole-vertebra mask into body / posterior elements; return the BODY.

    Watershed on the distance transform, markers from the distance-thresholded cores
    (Yao 2006). `core_mm` is a THICKNESS, in mm, not a fraction of anything -- a pedicle
    isthmus is thinner than this and a vertebral body is thicker, at every level and in
    every patient, which is why this generalises where a fixed fraction does not.

    Returns a boolean array, or None if the partition is not possible (a single core, a
    mask too small to be a vertebra). FALLBACK ONLY -- see body_mask.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    sp = voxel_spacing(affine)
    edt = ndimage.distance_transform_edt(m, sampling=sp)
    cores, n = ndimage.label(edt >= float(core_mm))
    if n == 0:
        return None
    # drop specks so a stray thick voxel cannot seed a basin
    sizes = ndimage.sum(np.ones_like(cores), cores, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_core_vox]
    if not keep:
        return None
    if len(keep) == 1:                       # nothing to separate: body only, or fused
        return m
    markers = np.zeros_like(cores)
    for new, old in enumerate(keep, start=1):
        markers[cores == old] = new
    from skimage.segmentation import watershed
    basins = watershed(-edt, markers=markers, mask=m)
    # the BODY is the basin holding the global thickness maximum -- a vertebral body is
    # the thickest part of a vertebra whether or not the arch is complete
    body_label = int(basins[np.unravel_index(np.argmax(edt * (basins > 0)), edt.shape)])
    if body_label == 0:
        return None
    return basins == body_label


def fit_plane_ransac(points, *, thresh_mm: float = 1.5, iters: int = 200,
                     min_inlier_frac: float = 0.5, seed: int = 0
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Robust plane fit by RANSAC (Fischler & Bolles 1981).

    Returns (centroid, unit normal, inlier boolean mask), or None. Unlike iterative
    trimming this has a breakdown point: gross outliers -- an osteophyte lip, a
    segmentation spur -- cannot move the consensus as long as the endplate itself is the
    largest coherent planar set, which it is.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        return None
    rng = np.random.default_rng(seed)
    best_in, best_n, best_c = None, None, None
    for _ in range(int(iters)):
        idx = rng.choice(len(P), size=3, replace=False)
        a, b, c = P[idx]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        d = np.abs((P - a) @ nrm)
        inl = d <= thresh_mm
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_n, best_c = inl, nrm, a
    if best_in is None or best_in.sum() < max(3, min_inlier_frac * len(P) * 0.2):
        return None
    # refit on the consensus set by total least squares
    Q = P[best_in]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    return c, unit(vt[-1]), best_in


def endplate_surface_body(body_pts, which: str, *, sup_axis=WORLD_SUPERIOR,
                          lr=(1.0, 0.0, 0.0), nbins: int = 26) -> np.ndarray:
    """Disc-facing cortical surface of an ISOLATED body: the extreme voxel along the
    superior axis in each (A-P, L-R) cell.

    No posterior-element rejection here -- `body_pts` has none. That is the whole point
    of doing the partition first.
    """
    P = np.asarray(body_pts, float)
    if len(P) < 6:
        return P
    a = unit(sup_axis)
    lrv = unit(lr)
    ap = anterior_axis(a, lr)
    sgn = 1.0 if which == "superior" else -1.0
    sc = (P - P.mean(0)) @ ap
    lc = (P - P.mean(0)) @ lrv
    si = np.floor((sc - sc.min()) / (np.ptp(sc) + 1e-9) * nbins).astype(int)
    li = np.floor((lc - lc.min()) / (np.ptp(lc) + 1e-9) * nbins).astype(int)
    key = si * (nbins + 1) + li
    order = np.lexsort((-sgn * (P @ a), key))
    ks = key[order]
    first = np.ones(len(order), bool)
    first[1:] = ks[1:] != ks[:-1]
    return P[order[first]]


def endplate_corners_body(mask, affine, which: str = "superior", *,
                          sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                          core_mm: float = 4.0, ransac_mm: float = 1.5,
                          lat_frac: float = 0.45, min_points: int = 30,
                          canal_from=None):
    """Endplate corners of ONE vertebra, from its whole-vertebra mask.

    body_mask -> medial band -> endplate surface -> RANSAC plane -> A-P extremes of the
    inliers. Returns (anterior, posterior, rms, n_inliers) in world mm, or None.

    `rms` is the residual of the endplate surface about the RANSAC plane and is the
    honest confidence on this endplate -- a level whose plate did not converge should be
    dropped, not drawn. The medial band is kept (it drops the transverse processes and,
    on the sacrum, the alae, which are part of the body's own component and so survive
    the watershed) but every other tuning parameter of the old path is gone.
    """
    m = np.asarray(mask, bool)
    # pass the CALLER's axes through -- defaulting silently to the world axes made this
    # correct only when the caller happened to use them, which a test harness does and
    # the DRR path (detector cranial axis, beam direction) does not
    body = body_mask(m, affine, sup_axis=sup_axis, lr=lr, core_mm=core_mm,
                     canal_from=canal_from)
    if body is None or body.sum() < min_points:
        return None
    idx = np.array(np.nonzero(body)).T
    pts = (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]
    lrv = unit(lr)
    if 0.0 < lat_frac < 1.0:
        lp = pts @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        pts = pts[(lp >= lo) & (lp <= hi)]
    if len(pts) < min_points:
        return None
    surf = endplate_surface_body(pts, which, sup_axis=sup_axis, lr=lr)
    if len(surf) < 6:
        return None
    fit = fit_plane_ransac(surf, thresh_mm=ransac_mm)
    if fit is None:
        return None
    c, n, inl = fit
    keep = surf[inl]
    if len(keep) < 4:
        return None
    rms = float(np.sqrt(np.mean(((keep - c) @ n) ** 2)))
    ap = anterior_axis(unit(sup_axis), lr)
    t = keep @ ap
    return keep[int(np.argmax(t))], keep[int(np.argmin(t))], rms, int(inl.sum())


def body_mask(mask, affine, *, sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
              core_mm: float = 4.0, canal_from=None) -> Optional[np.ndarray]:
    """The vertebral BODY: canal criterion, falling back to the watershed.

    Canal first because it is level-independent; watershed only when the arch does not
    enclose a canal (incomplete or lytic arch), where a thin-neck cut is the best
    available. None if neither succeeds -- callers should drop the level rather than
    measure a body that was never isolated.
    """
    b = body_mask_by_canal(mask, affine, sup_axis=sup_axis, lr=lr, canal_from=canal_from)
    if b is not None and b.any():
        return b
    return body_mask_watershed(mask, affine, core_mm=core_mm)
