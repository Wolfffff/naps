# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NAPS (NAPS is ArUco Plus SLEAP) quantifies animal behavior and tracks individuals with identity persistence. It combines pose-estimation tracks from SLEAP with fiducial markers (ArUco tags) decoded from the video. NAPS reads and writes SLEAP files through `sleap-io`; it does **not** import the full `sleap` package (TensorFlow) any more.

## Key Commands

Everything is configured in `pyproject.toml` (single source for packaging, ruff, ty, pytest, coverage).

### Installation
- Development install: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"` (Python >= 3.10)
- Users: `pip install naps-track` or `conda create -n naps -c kocherlab -c conda-forge naps-track`
- Extras: `dev` (pytest, pytest-cov, ruff, ty, pre-commit, build, twine, validate-pyproject), `docs` (Sphinx). No SLEAP extra on purpose: `sleap` pulls in `opencv-python`, which clobbers `opencv-contrib-python-headless`.

### Testing
- All tests: `pytest` (testpaths, strict markers and warnings-as-errors are set in `pyproject.toml`)
- With branch coverage: `pytest --cov --cov-report=term-missing` (`--cov` takes its source and branch setting from `[tool.coverage]`; `fail_under` applies)
- One file: `pytest tests/test_aruco.py`

### Linting, formatting, type checking
- Lint: `ruff check .` (autofix: `ruff check --fix .`)
- Format: `ruff format .` (check only: `ruff format --check .`)
- Types: `ty check` (run inside the environment that has the project installed; ty finds it through `VIRTUAL_ENV`)
- All of them on commit: `pre-commit install`, then `pre-commit run --all-files`

### Packaging
- Build: `python -m build` (or `uv build`), then `twine check dist/*`
- The version lives only in `naps/__init__.py::__version__` (setuptools reads it with `attr:`); do not duplicate it.
- Releases: bump `__version__`, update `CHANGELOG.md`, tag `vX.Y.Z`, publish a GitHub release. `python-publish.yml` uses PyPI trusted publishing (OIDC) and `upload_conda.yml` uploads the noarch conda package built from `.conda/naps-track/meta.yaml`.

### Main Entry Points (console scripts, defined in `[project.scripts]`)
- `naps-track`: main tracking command (`naps/naps_track.py:main`; `build_parser()` is what the Sphinx docs render)
- `naps-plot`: model overlays on video (`naps/naps_plot.py`)
- `naps-interactions`: interaction assignment (`naps/naps_interactions.py`)

## Architecture Overview

### Core Modules
- **naps/aruco.py**: `ArUcoModel` wraps `cv2.aruco` (OpenCV >= 4.7 `ArucoDetector`, with a fallback for older builds); optional decoder subset
- **naps/matching.py**: `Matching` coordinates marker detection with SLEAP tracks and the cost matrix
- **naps/cost_matrix.py**: rolling-window cost matrix and Hungarian assignment (weighted votes); the window state lives in **naps/rolling_votes.py** (same row/column order as the old DataFrame code, so tie-breaking is unchanged; `tests/legacy_cost_matrix.py` is the oracle)
- **naps/decode.py**, **naps/video.py**, **naps/crops.py**: the decode stage (`TagDecoder`, `decode_video`, `FastVideoReader`, `split_range`, `open_capture`); the numbers behind the presets are in `docs/decode_measurements.md`
- **naps/parallel_decode.py**: chunked parallel decode (`run_chunks`, `decode_in_chunks`), used through `matching.ChunkedDecodeSource` (`--decode-workers`); the merged result must equal the serial one
- **naps/slp_tracks.py**, **naps/slp_fast.py**: direct HDF5 read of the tag positions and a vectorised `.slp` writer with the stock writer as fallback (`--no-fast-read`, `--no-fast-write`)
- **naps/evidence.py**: `EvidenceSource` ABC, `ArucoEvidence`, `IdtrackerEvidence`, `collect_votes`
- **naps/closed_set.py**: closed-set elimination (`eliminate_identities`)
- **naps/quadrants.py**: quadrant-aware closed set (regions, `QuadrantScope`, inference from decodes, per-region elimination, `--report-json`)
- **naps/tags.py**: `normalize_tag` / `tag_name`; tags are plain ints and tracks are named `ArUcoTag#<int>`
- **Joint identity engine** (opt-in `--identity-engine joint`; the legacy path is unchanged):
  - **naps/instances.py**: `InstanceTable` (padded per-frame arrays of every SLEAP instance), `table_from_labels`
  - **naps/tracklets.py**: own tracklet builder (position/velocity/pose gates, competing-instance breaks, SLEAP ids as hints)
  - **naps/identity.py**: exact forward-backward over injective tag assignments per region (N <= 6), anchors, stray demotion, `solve_region`
  - **naps/consistency.py**: temporal-consistency stage (ghosts, duplicates, chain veto, gated interpolation), calibrated `ConsistencyParams`
  - **naps/diagnostics.py**: window status, completeness/sources/abstentions/jumps report, hold-out self-validation
  - **naps/joint.py**: `JointMatching` (decode every instance, split into regions, solve, scatter), sidecar writer
  - **naps/sources.py**, **naps/reasons.py**: label source codes and abstention reason codes (part of the sidecar format, schema version in `naps/joint.py`; never renumber, append only)
  - Pipeline stages, sidecar schema v2, recommended profile and measured limits: `docs/identity_pipeline.md` (keep it in sync with `--min-confidence` and `reasons.py`)
- **naps/sleap_utils.py**: SLEAP I/O helpers on top of `sleap_io`
- **naps/utils/**: `tracking.py` (TrackingArray, used by `naps-plot`/`naps-interactions`) and `interactor.py`

### Data Flow
1. Video + SLEAP predictions (`.slp` / analysis `.h5`) are loaded
2. ArUco markers are decoded around each tag node with OpenCV
3. Votes (ArUco, optionally idtracker.ai) are accumulated per track in a rolling window and assigned with the Hungarian method
4. Optional closed-set elimination fills in the last missing tag of a colony or quadrant
5. Tracks are renamed `ArUcoTag#<id>` and saved as a SLEAP analysis file

### Dependencies
Runtime dependencies are listed once, in `pyproject.toml` (`[project].dependencies`): numpy, scipy, pandas, h5py, attrs, pyyaml, `opencv-contrib-python-headless`, `sleap-io`, matplotlib, shapely. Keep `.conda/naps-track/meta.yaml` in sync when they change. Only one OpenCV distribution may be installed.

## Conventions
- Python >= 3.10, line length 100, Google-style docstrings, ruff rules as set in `pyproject.toml` (tests are exempt from docstring/annotation rules).
- Everything new in `naps-track` must be opt-in: default behaviour must stay identical to the committed golden test.
- Public camelCase names in `naps/` are API; keep them and mark the definition with `# noqa: N802` / `N803` instead of renaming.
