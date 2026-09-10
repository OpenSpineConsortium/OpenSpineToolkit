# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

OpenSpineToolbox is a teaching/research repo of independent student miniprojects
that compute spinopelvic measurements (Cobb angle, pelvic incidence, lordosis,
etc.) from CTSpinoPelvic1K v3 CT segmentation masks. Two things coexist here:

- **`ostk/`** — a real, tested Python package of shared geometry primitives and
  fully-implemented measurements (PI/SS/PT, lumbar lordosis, PI-LL mismatch +
  SRS-Schwab modifiers, coronal Cobb angle, and per-level
  morphometry in each vertebra's own frame). This is where Claude will do most
  substantive work.
- **`projects/<name>/`** — one folder per student miniproject (some are just a
  README spec waiting to be implemented; a few have a `main.py`). Each project
  is meant to reuse `ostk` primitives rather than re-deriving geometry.

**`SPEC.md` is the authoritative technical contract** — read it before touching
measurement code. It defines the v3 label-id scheme, the world-mm/affine
convention, the shared geometry primitives, the output JSON contract, and which
parameters are valid on supine CT vs. out of scope (SVA/TPA need C7/T1, which v3
doesn't have). `README.md` is only Git/PR mechanics for student contributors —
not architecture.

## Commands

```bash
# run the full test suite
python -m pytest

# run a single test file / test
python -m pytest tests/test_metrics.py
python -m pytest tests/test_metrics.py::test_pelvic_incidence_identity -q

# run the CLI over a directory of NIfTI label maps
python -m ostk pi   --labels labels/ --out pi.csv       # pelvic incidence
python -m ostk ll   --labels labels/ --out ll.jsonl      # lumbar lordosis
python -m ostk cobb --labels labels/ --out cobb.csv      # coronal Cobb angle
python -m ostk all  --labels labels/ --out summary.csv --workers 8   # full spinopelvic summary
python -m ostk morph --labels labels/ --out morph.csv --workers 8  # per-level dimensions

# with no --out, one-line-per-case JSON is printed to stdout instead
```

There is no build/lint step and no package install (`pytest.ini` sets
`pythonpath = .`, so `ostk` imports directly from the repo root). Dependencies
are `numpy`, `scipy`, `nibabel`, `pytest` (`requirements.txt`); tests that need
`nibabel` use `pytest.importorskip("nibabel")` so the geometry-only tests still
run without it.

## Architecture of `ostk/`

Everything is layered so higher levels never hard-code magic numbers or
re-derive geometry:

```
labels.py    label-id scheme (LABELS dict, lid()) — never hard-code an id, use lid("L1")
io.py        NIfTI load + voxel_to_world (nibabel imported lazily; geometry stays pure)
geometry.py  stateless world-mm primitives: PCA (principal_axes), TLS plane fit,
             sphere fit, angle/cobb_angle helpers — no numpy RNG, sign-fixed
             eigenvectors, so results are deterministic and picklable
masks.py     binary_mask / largest_component / endplate_points over label volumes
spine.py     fit_endplate — the vertebral-body endplate primitive (anterior-filter
             to drop posterior elements, then extreme-voxel-per-column to
             respect tilt). Shared by metrics.py and cobb.py; this is the file to
             improve if endplate fidelity needs work.
metrics.py   composed measurements: pelvic_incidence, lumbar_lordosis,
             pi_ll_mismatch, schwab_sagittal_modifiers, and the
             *_from_label wrappers that go straight from a label volume to a
             Measurement
cobb.py      coronal Cobb angle; reuses spine.fit_endplate and
             metrics._lr_axis_from_label rather than re-deriving the L-R axis
record.py    Measurement dataclass — the one output contract (SPEC §4)
parallel.py  map_cases() — runs a *_from_label worker over many files via
             ProcessPoolExecutor; workers must stay top-level functions to pickle
cli.py       argparse entry point (python -m ostk <pi|ll|cobb|all>)
```

**Key conventions to preserve when adding a measurement:**

- All geometry is in **world millimetres via the NIfTI affine** — never voxel
  index space (`ostk.io.voxels_to_world`).
- The **patient sagittal plane** is derived from data, not scanner axes: the
  left-right axis is the vector between the two femoral-head centres
  (`metrics._lr_axis_from_label`), so results are robust to patient roll/tilt.
  The coronal (anterior) axis for Cobb work is `spine.anterior_axis`, built
  from that same L-R vector — don't derive a second, inconsistent L-R estimate.
- Every `*_from_label` function returns a `record.Measurement` (or, for the
  multi-parameter summary, a plain dict following the same shape) and **never
  silently drops a bad case** — missing/small inputs produce `value=None` plus
  a `qc_flags` entry (`missing_label:<id>`, `low_voxels:<id>`,
  `fit_residual_high:<which>`, `identity_violation`, etc.), not an exception.
- Mark supine-CT-derived values with `supine_ct=True` and treat SS/PT/LL/Cobb
  as supine surrogates, not standing-equivalent values (SPEC §5's golden rule).
- Sign/orientation ambiguities (PCA eigenvectors, endplate normals) are fixed
  deterministically (see `geometry._orient`, `spine.fit_endplate`'s cranial
  orientation) — repeated runs must produce identical output.
- Worker functions passed to `parallel.map_cases` / used in `cli.py` must stay
  top-level (module-level) functions so they can be pickled across processes.

## Working within `projects/`

Each miniproject folder is independent and self-contained (its own
`README.md` + `main.py`); they are not imported by `ostk` or by each other.
When implementing or extending one, prefer importing primitives from `ostk`
(e.g. `ostk.spine.fit_endplate`, `ostk.geometry.cobb_angle`) over duplicating
geometry code, matching how `projects/scoliosis-cobb-angle/` and
`projects/sacral-slope-pelvic-incidence/` do it.
