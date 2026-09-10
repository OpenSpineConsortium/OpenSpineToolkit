"""ostk.labels -- the label-id schemes ostk can read, and which one a volume uses.

THERE IS MORE THAN ONE SCHEME, AND MIXING THEM DOES NOT RAISE. That is the whole
reason this module exists in its current form. The legacy ostk scheme puts L1 at 1
and the femurs at 11/12; the published CTSpinoPelvic1K release (v10, Zenodo
10.5281/zenodo.22642578) is VerSe-native and puts L1 at 20 and the femurs at 32/33.
Resolving "femur_left" against the wrong map is not an error -- it returns id 11,
which in a released volume is T4, and a sphere is then fitted to a thoracic vertebra.
The measurement comes back finite, plausible-looking and wrong. That failure shipped:
pelvic incidence was computed for the entire release against the legacy map, and the
only visible symptom was a QC flag saying S1 had too few voxels -- while S1 sat in the
volume with two hundred thousand.

So nothing here guesses. `detect_scheme` reads the volume and decides, `labels_for`
hands back the matching map, and every public `*_from_label` entry point resolves
through them. When the evidence does not clearly favour one scheme, detection RAISES
rather than picking the more likely one: a loud failure costs a re-run, a silent one
costs a paper.

    from ostk.labels import labels_for, lid
    L = labels_for(volume)            # {name: id} for THIS volume
    sacrum = volume == L["sacrum"]

`lid(name)` without a volume still resolves against the default scheme for callers
that know what they hold; it is kept for compatibility and is the thing to avoid in
new code.
"""
from __future__ import annotations

import numpy as np

# --- the legacy ostk scheme (CTSpinoPelvic1K v3/v4, cores 0-49 + soft tissue) -------
# v3 populates 0-49 (cores + femurs + GT thoracic + TS ribs) with ignore 50, and
# RESERVES the v4 soft-tissue block: iliolumbar (51/52), LS-nerve roots (53-58),
# psoas (59/60). v4 populates that block and relocates ignore 50 -> 255.
LABELS_V4 = {
    "background": 0,
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
    "S1": 7, "sacrum": 8, "left_hip": 9, "right_hip": 10,
    "femur_left": 11, "femur_right": 12,
    **{f"T{n}": 12 + n for n in range(1, 14)},           # T1..T13 -> 13..25
    **{f"rib_left_{n}": 25 + n for n in range(1, 13)},   # 26..37
    **{f"rib_right_{n}": 37 + n for n in range(1, 13)},  # 38..49
    "iliolumbar_left": 51, "iliolumbar_right": 52,
    "nerve_L4_left": 53, "nerve_L4_right": 54,
    "nerve_L5_left": 55, "nerve_L5_right": 56,
    "nerve_S1_left": 57, "nerve_S1_right": 58,
    "psoas_left": 59, "psoas_right": 60,                 # v4 (XLIF corridor)
    "aorta": 61, "inferior_vena_cava": 62,               # v4 great vessels
    "iliac_artery_left": 63, "iliac_artery_right": 64,
    "iliac_vena_left": 65, "iliac_vena_right": 66,
}

# --- the PUBLISHED scheme (CTSpinoPelvic1K v10, VerSe-native, contiguous 0-68) ------
# Mirrors scripts/label_scheme.py in the dataset repo and dataset_labels.json in the
# release itself; both are the source of truth and this must not drift from them.
# Bone and hardware only -- no soft tissue. Ribs run 13 per side because a thirteenth
# rib is a finding, not a mislabelled twelfth.
LABELS_V10 = {
    "background": 0,
    **{f"C{n}": n for n in range(1, 8)},                  # C1..C7 -> 1..7
    **{f"T{n}": 7 + n for n in range(1, 13)},             # T1..T12 -> 8..19
    "L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24, "L6": 25,
    "sacrum": 26, "coccyx": 27, "T13": 28, "S1": 29,
    "left_hip": 30, "right_hip": 31, "femur_left": 32, "femur_right": 33,
    **{f"rib_left_{n}": 33 + n for n in range(1, 14)},    # 34..46 (13 = T13 rib)
    **{f"rib_right_{n}": 46 + n for n in range(1, 14)},   # 47..59
    "rib_left_lumbar": 60, "rib_right_lumbar": 61,
    "hardware": 62, "hardware_cage": 63, "hardware_screw_rod": 64,
    "hardware_plate": 65, "hardware_arthroplasty": 66,
    "hardware_si_screw": 67, "hardware_osteosynthesis": 68,
}

SCHEMES = {"v4": LABELS_V4, "v10": LABELS_V10}
DEFAULT_SCHEME = "v10"          # the published release; what ostk is documented against

# Backwards compatibility: `LABELS` was the legacy map and code still imports it.
LABELS = LABELS_V4
ID_TO_NAME = {v: k for k, v in LABELS_V4.items()}

LUMBAR = ("L1", "L2", "L3", "L4", "L5", "L6")
THORACIC = tuple(f"T{n}" for n in range(1, 14))
IGNORE_V3 = 50
IGNORE_V4 = 255

# WHAT SEPARATES THE TWO, MEASURED RATHER THAN ASSUMED. The pelvis is the discriminator
# because hips and femurs are the largest objects either scheme names, and the two
# schemes put them at completely different ids: 9-12 in v4, 30-33 in v10. In a released
# volume ids 9-12 are T2-T5, which an abdominal field of view usually does not even
# contain; in a legacy volume ids 30-33 are mid ribs, which are present but two orders
# of magnitude smaller than a femur. Counting voxels under each hypothesis therefore
# separates them by a wide margin instead of by a hair.
_PELVIS = ("left_hip", "right_hip", "femur_left", "femur_right")
_MARGIN = 4.0            # winner must claim this many times the loser's pelvis voxels


class SchemeError(RuntimeError):
    """Raised when a volume's label scheme cannot be determined with confidence."""


def _pelvis_voxels(counts: dict, scheme: str) -> int:
    m = SCHEMES[scheme]
    return int(sum(counts.get(m[n], 0) for n in _PELVIS))


def detect_scheme(volume, *, counts=None) -> str:
    """Which scheme `volume` is labelled in: "v4" or "v10".

    Decided on pelvis voxel mass under each hypothesis (see above). Three outcomes,
    and the split between them is the whole point:

      NO PELVIS UNDER EITHER -- an empty volume, or a spine-only crop. There is no
        scheme to get wrong here, because every subsequent lookup finds nothing
        whichever map is used and the caller returns None through its ordinary path.
        Returns the default rather than raising, so a legitimately empty case stays a
        None instead of becoming an exception the caller has to know about.
      BOTH PLAUSIBLE, CLOSE TOGETHER -- genuinely ambiguous, and picking would be a
        guess with a wrong bone at the end of it. RAISES.
      A CLEAR WINNER -- returned.
    """
    if counts is None:
        ids, n = np.unique(np.asarray(volume), return_counts=True)
        counts = dict(zip(ids.tolist(), n.tolist()))
    v4, v10 = _pelvis_voxels(counts, "v4"), _pelvis_voxels(counts, "v10")
    hi, lo = max(v4, v10), min(v4, v10)
    if hi == 0:
        return DEFAULT_SCHEME
    if lo and hi < _MARGIN * lo:
        raise SchemeError(
            f"scheme is ambiguous: v4 pelvis {v4} voxels vs v10 pelvis {v10}, "
            f"within the {_MARGIN}x margin; pass scheme= explicitly")
    return "v10" if v10 > v4 else "v4"


def labels_for(volume=None, *, scheme: str | None = None, counts=None) -> dict:
    """The {name: id} map for `volume`, detected unless `scheme` is given."""
    if scheme is not None:
        if scheme not in SCHEMES:
            raise SchemeError(f"unknown scheme {scheme!r}; have {sorted(SCHEMES)}")
        return SCHEMES[scheme]
    if volume is None and counts is None:
        return SCHEMES[DEFAULT_SCHEME]
    return SCHEMES[detect_scheme(volume, counts=counts)]


def id_to_name_for(volume=None, *, scheme: str | None = None) -> dict:
    return {v: k for k, v in labels_for(volume, scheme=scheme).items()}


def lid(name: str, *, scheme: str | None = None) -> int:
    """Label id for a structure name in one scheme (raises on typo -- fail loud).

    Prefer `labels_for(volume)[name]` in new code: this resolves against the default
    scheme when none is given, and defaulting is precisely what went wrong before.
    """
    return labels_for(scheme=scheme or DEFAULT_SCHEME)[name]
