"""The label scheme is detected, not assumed.

These exist because the opposite shipped. Pelvic incidence was computed for an entire
802-record release with the legacy map against VerSe-native volumes, so "S1" resolved to
id 7 -- which in a released volume is C7 -- and "femur_left" to id 11, which is T4. The
run did not error. Every case came back with a QC flag saying S1 had too few voxels while
S1 sat in the volume with two hundred thousand, and the only reason it was caught was that
the numbers disagreed with a previous run.
"""
import numpy as np
import pytest

from ostk.labels import (LABELS_V4, LABELS_V10, DEFAULT_SCHEME, SchemeError,
                         detect_scheme, labels_for, lid)


def _phantom(scheme: str, pelvis_voxels: int = 4000, D: int = 40):
    """A volume with a pelvis and a lumbar spine written in `scheme`'s ids."""
    L = {"v4": LABELS_V4, "v10": LABELS_V10}[scheme]
    v = np.zeros((D, D, D), np.int16)
    k = int(round(pelvis_voxels ** (1 / 3)))
    for i, name in enumerate(("left_hip", "right_hip", "femur_left", "femur_right")):
        v[2 + i * 6: 2 + i * 6 + k, 2:2 + k, 2:2 + k] = L[name]
    for i, name in enumerate(("L1", "L2", "L3", "L4", "L5")):
        v[20:24, 20:24, 5 + i * 5: 9 + i * 5] = L[name]
    return v


def test_detects_each_scheme_from_its_own_volume():
    assert detect_scheme(_phantom("v4")) == "v4"
    assert detect_scheme(_phantom("v10")) == "v10"


def test_labels_for_returns_the_matching_map():
    assert labels_for(_phantom("v10"))["S1"] == 29
    assert labels_for(_phantom("v4"))["S1"] == 7
    # and the femurs, which is where the shipped bug did its damage
    assert labels_for(_phantom("v10"))["femur_left"] == 32
    assert labels_for(_phantom("v4"))["femur_left"] == 11


def test_the_shipped_bug_would_now_be_caught():
    """A released volume must not resolve "S1" to the legacy id 7 (=C7)."""
    v10 = _phantom("v10")
    assert labels_for(v10)["S1"] != LABELS_V4["S1"]
    assert (v10 == labels_for(v10)["femur_left"]).sum() > 0      # a real femur
    assert (v10 == LABELS_V4["femur_left"]).sum() == 0           # what it used to read


def test_explicit_scheme_overrides_detection():
    v = _phantom("v10")
    assert labels_for(v, scheme="v4")["S1"] == 7
    with pytest.raises(SchemeError):
        labels_for(v, scheme="v11")


def test_empty_volume_does_not_raise():
    """Nothing to get wrong: every lookup finds zero voxels either way, and the caller
    returns None through its ordinary path rather than through an exception."""
    assert detect_scheme(np.zeros((8, 8, 8), np.int16)) == DEFAULT_SCHEME


def test_ambiguous_volume_raises_rather_than_guessing():
    """Both hypotheses find a comparable pelvis: picking one would be a coin flip with
    a wrong bone at the end of it."""
    # Both pelvises must actually COEXIST, so they are written to disjoint corners --
    # overlaying two phantoms at the same coordinates just lets one overwrite the other.
    both = _phantom("v10", D=48)
    w = _phantom("v4", D=48)
    both[24:, :, :] = w[24:, :, :]
    both[:24, 20:24, :] = _phantom("v10", D=48)[:24, 20:24, :]
    v4n = sum((both == LABELS_V4[n]).sum() for n in
              ("left_hip", "right_hip", "femur_left", "femur_right"))
    v10n = sum((both == LABELS_V10[n]).sum() for n in
               ("left_hip", "right_hip", "femur_left", "femur_right"))
    assert v4n and v10n and max(v4n, v10n) < 4 * min(v4n, v10n), (v4n, v10n)
    with pytest.raises(SchemeError):
        detect_scheme(both)


def test_lid_still_resolves_for_callers_that_know_what_they_hold():
    assert lid("L1", scheme="v4") == 1
    assert lid("L1", scheme="v10") == 20
    assert lid("L1") == LABELS_V10["L1"]          # default is the published release


def test_the_two_schemes_actually_differ_where_it_matters():
    for n in ("L1", "L5", "S1", "sacrum", "left_hip", "femur_left", "femur_right"):
        assert LABELS_V4[n] != LABELS_V10[n], f"{n} collides; detection cannot help"
