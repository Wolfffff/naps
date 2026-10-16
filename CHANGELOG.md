# Changelog

All notable changes to NAPS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The version is defined in `naps/__init__.py`.

## [Unreleased]

Everything below is relative to 1.0.9 (`main`). Every new tracking option is opt-in: without the new
flags `naps-track` runs the same pipeline as before (the committed golden-output test still passes),
apart from the fixes listed under "Fixed".

### Added

- **Joint identity engine** (`--identity-engine joint`, opt-in; default stays `legacy`): builds tracklets from the
  SLEAP instances itself (`naps.tracklets`; `--tracklets built|sleap`), decodes the tag around every instance and
  infers all tags of a colony/quadrant jointly with an exact forward-backward over the injective assignments
  (`naps.identity`; closed set of N <= 6 tags, 3 tags + 1 untagged bee supported), then removes implausible recovered
  labels (`naps.consistency`: ghost instances, duplicate tags, chain veto, gated interpolation; `--consistency`,
  `--interpolate-max-gap`) and reports per region and window (`naps.diagnostics`: completeness before/after,
  sources, abstentions, disputes, jumps, window status, optional hold-out `--self-validate`). Clean, unique decodes are
  immutable anchors (a contradicted decode is only flagged `disputed`); a decode that is contradicted (the same tag on
  two instances of a frame) or a minority tag on its tracklet is demoted to weak evidence, is counted in the report and
  the instance may be relabelled. The first and last frames are labelled, labels below `--min-confidence` are not
  written, outputs carry `identity_score` and a per-frame sidecar (`--identity-sidecar`), tag ids are
  session-qualified (`--session`).
  New modules: `naps/instances.py`, `naps/tracklets.py`, `naps/identity.py`, `naps/consistency.py`,
  `naps/diagnostics.py`, `naps/joint.py`, `naps/sources.py`, `naps/reasons.py`; `naps.matching` gains the `DecodeSource` protocol
  (`OpenCVDecodeSource`, `TagDecoderSource`; `--decoder legacy|fast`) and `naps.sleap_utils.apply_instance_tags`
  writes instance-keyed labels. The legacy path is untouched.
- **Identity sidecar schema version 2** (`schema_version` array in `<output>.identity.npz`; files without it are
  version 1): a per-instance `reason` code for every instance with a position that ends up without a tag (stable
  enum `naps.reasons.Reason`, names in `reason_names`: ghost, duplicate, excess, below-min-confidence, low confidence
  in a pile, no anchor in the tracklet, untagged bee, blocked by anchor, consistency-dropped, withheld, no position,
  outside the regions, region without data / with an unsupported tag set), a per-instance `tracklet_id` (unique
  across the regions, `tracklet_offset__<region>`) and `decode_demoted`. `--report-json` gains
  `abstentions_by_reason`, `demoted_decodes`, `reason_names`, `n_tracklets` and `sidecar_schema_version`;
  `RegionLabels.info` gains `n_conflict_decodes_demoted`, `n_demoted_decodes_relabelled`,
  `n_demoted_decodes_confirmed` and `n_ghost_instances_culled`. No `--tracklets-only` mode: the tracklets depend on the
  decodes (anchored instances rank first when a frame holds more instances than bees, the ghost test needs them), so
  they cannot be built before decoding, which dominates the run time.
- **CPU speed work for `naps-track`; none of it changes the result** (the tags, every dataset of the output `.slp`
  and the `--report-json` are identical to the previous code; the raw `.slp` bytes are not stable even between two
  runs of the same code because `sleap-io` writes the skeleton's symmetry edges in set order):
  - the cost matrix keeps its rolling window as an ordered dictionary plus a small array
    (`naps/rolling_votes.py`) instead of building a `pandas` DataFrame per frame, in the same row and column order so
    that `scipy`'s tie-breaking is unchanged; the previous assignment is reused when the array is identical. A
    verbatim copy of the old class (`tests/legacy_cost_matrix.py`) is the oracle of seeded differential tests. The
    DataFrame cost was 38-57 % of the wall time of an hour-long run on the lab's videos.
  - `.slp` I/O: the tag-node positions are read straight from the file's HDF5 tables (`naps/slp_tracks.py`) and the
    output is written by a vectorised writer (`naps/slp_fast.py`, used only for the sleap-io versions it was verified
    against, with the stock writer as fallback); `--no-fast-read` / `--no-fast-write` switch them off.
  - `--decode-workers N`, `--decode-chunk-frames`, `--decode-gop`, `--decode-gop-offset`: the video decode runs in
    parallel over contiguous frame ranges (`naps/parallel_decode.py`, `ChunkedDecodeSource`; works with both identity
    engines and both decoders), each worker with one FFmpeg decoder thread (`naps.video.open_capture`, a
    `threads=` option of `FastVideoReader`); `naps.video.split_range` gained `align=` / `offset=` so chunks start near
    keyframes. The merged decodes equal the serial ones (seam tests); the SLEAP file is loaded in a background thread
    during the decode. Defaults are unchanged (serial).
- `docs/decode_measurements.md`: the benchmark numbers quoted in the decode docstrings (Beszel Trial2 camera0 and the
  12-colony `minMarkerPerimeterRate` study).
- **Closed-set tags** (`--closed-set ID [ID ...]`): the N tags expected in the video. ArUco votes for any
  other tag are dropped and closed-set elimination runs: in a frame where exactly one expected tag and
  exactly one tracked instance are unassigned, the instance receives that tag, provided it lies within
  `--closed-set-max-jump` pixels per frame of gap of where that tag was last or next seen
  (`--closed-set-max-distance` caps the threshold in absolute pixels). `--no-closed-set-elimination`
  keeps only the vote filter. `--aruco-marker-subset ID [ID ...]` restricts the ArUco decoder itself to a
  subset of the dictionary.
- **Quadrant-aware closed set** for full frames holding several independent colonies:
  `--closed-set-map` (inline `"top_left:45,46,47,48;..."` or a `.json`/`.yaml` file) or
  `--closed-set-infer` (infer the tag sets from the video's own ArUco decodes; ambiguous regions abstain
  and keep the legacy behaviour), with `--closed-set-size`, `--closed-set-min-support`,
  `--closed-set-min-fraction`, `--closed-set-min-purity`, regions from `--quadrant-grid ROWS COLS` or
  repeatable `--quadrant-roi NAME X0 Y0 X1 Y1`, and `--position-node-name` (default `Thorax`) to place
  instances whose tag node is NaN. Mutually exclusive with `--closed-set`. Elimination runs per region
  with a 400 px continuity cap by default.
- **idtracker.ai continuity evidence** (`--idtracker-trajectories PATH`, with `--idtracker-offset`,
  `--idtracker-frame-offset`, `--idtracker-weight`, `--idtracker-min-prob`, `--idtracker-max-dist`,
  `--idtracker-min-margin`, `--idtracker-min-support`, `--idtracker-min-purity`). ArUco decodes stay the
  identity source; idtracker.ai only votes for tracks without an ArUco decode, after its arbitrary labels
  are aligned to tags by co-occurrence. Interpolated positions (NaN `id_probability`) never vote.
- `--report-json PATH`: per-video diagnostics (frames with 4/3/2/1/0 tags per region before and after
  elimination, untagged instances by cause, eliminated assignments, abstention reasons, inferred tag sets,
  wrong-region votes). Works in every mode.
- `--strict-carry-forward`: only carry the previous frame's tag into tracks the linear assignment left
  unmatched. Off by default because it flipped tags 2-8x more often than the legacy sticky carry-forward
  on real data.
- Python API: `naps.evidence` (`EvidenceSource`, `ArucoEvidence`, `IdtrackerEvidence`, `collect_votes`),
  `naps.closed_set` (`eliminate_identities`), `naps.quadrants` (regions, `QuadrantScope`,
  `eliminate_by_quadrant`, `build_report`), `naps.tags` (`normalize_tag`, `tag_name`), and the new
  `Matching` arguments `evidence_sources`, `closed_set`, `eliminate`, `elimination_max_jump`,
  `elimination_max_look`, `elimination_max_distance`, `untracked_counts`, `quadrants`, `position_dict`,
  `untracked_positions` and `strict_carry_forward`.
- **Precomputed decodes** (`--decodes PATH`, opt-in): serve the tag votes from an `.npz` written by a pipeline
  that decoded the video once (for example a GPU pipeline feeding SLEAP and the tag decoder from one NVDEC
  decode) instead of re-reading the video. New module `naps/precomputed.py` (`PrecomputedDecodeSource`, a
  `DecodeSource`; `DecodesError`; the file layout is in its docstring); `Matching` gains `decode_source`
  (the video need not exist then). Works with `--identity-engine legacy` and `joint`; mutually exclusive with
  `--decoder fast`; a file that does not cover the frame range or does not match the `.slp` (instance counts,
  tag-node positions) exits with an error.
- **Detector speed knob `minMarkerPerimeterRate`**: `--aruco-min-marker-perimeter-rate RATE` (default none = unchanged)
  and `DecodeParams.min_marker_perimeter_rate` (default `None` = OpenCV's 0.03, the legacy decoder is untouched).
  OpenCV discards contours shorter than `RATE x max(crop width, height)` pixels; 0.03 is 3 px on a 100 px crop and
  lets many tiny noise quads reach the marker test. `naps.decode.FAST_MIN_MARKER_PERIMETER_RATE = 0.15`
  is now used by `CLOSED_SET_PASSES` / `TagDecoder.for_closed_set(..., min_marker_perimeter_rate=0.15)` and by
  `--decoder fast` (when the flag is not given). Evidence (53,640 real crops, 61 windows, 12 colony-trials x both
  cameras, hours 0-112, separated and huddled): no crop changes (0 lost, 0 gained, identical vote lists, unchanged
  wrong-in-set and decoy false-accept rates) for rates 0.10-0.30 with the legacy decoder, a single closed-set
  pass and the 3-pass preset at crop half-widths 50/64/100 and x1.5/x2 upsampling; 16,137 votes of `decode_video`
  on full frames identical. Per crop: legacy decoder 6.8x faster at half 50, single closed-set pass 1.7x, 3-pass
  preset about 2x, 1.0-1.6x at 200 px crops or x2 upsampling. Recall falls only when `RATE x crop side` exceeds the
  smallest tag's perimeter (96 px: 0.8 on a 128 px crop, 0.6 on a 200 px crop), so keep it below about 48 px
  (0.15 is safe up to 320 px crops). The rate is relative to the image: no rescaling is needed for `scale`.
- Packaging and tooling: `pyproject.toml` (PEP 621), `naps/py.typed`, ruff (lint and format), ty, pytest
  and coverage configuration, a pre-commit configuration and this changelog.
- Tests: a much larger suite (evidence, closed set, quadrants, `sleap_utils`, `naps-plot`,
  `naps-interactions`, `TrackingArray`, and end-to-end CLI tests on a synthetic 16-bee, four-quadrant
  file with real ArUco markers), with branch coverage enforced in `pyproject.toml`.

### Changed

- **`--min-confidence` help text and docstrings no longer claim that lower values are badly calibrated.** On 38
  held-out windows (396,323 bee-frames, scored against NAPS's own labels, not independent ground truth) the engine
  confidence was well calibrated (ECE 0.0028), so a lower threshold trades precision for coverage; 0.98 passed a
  pre-registered adoption rule under `--untrusted-policy emit`. The default stays 0.99. See
  `docs/identity_pipeline.md`, which also summarises the pipeline stages, sidecar schema v2, the recommended
  profile, the `--untrusted-policy` knife-edge and the known limits.
- **SLEAP I/O now goes through `sleap-io`**: `sleap` (and with it TensorFlow) is no longer imported by
  NAPS or required to run it. `get_location_matrix`, `load_tracks_from_slp` and `update_labeled_frames`
  now use `sleap_io.Labels`, and `update_labeled_frames` returns a `sleap_io.Labels`.
- **OpenCV 4.7+ / 5 ArUco API**: `ArUcoModel` uses `cv2.aruco.ArucoDetector`, `DetectorParameters()` and
  `getPredefinedDictionary` and falls back to the legacy functions on older builds. Marker ids are
  flattened correctly for both the `(N, 1)` (4.x) and `(N,)` (5.x) shapes.
- `CostMatrix` accepts weighted votes (a bare tag has weight 1, `(tag, weight)` or `None` are also
  accepted) and no longer leaves residue with fractional weights.
- `--aruco-crop-size` now defaults to 100 (the value used in the NAPS paper). Previously omitting it
  crashed with a `None` crop size.
- No existing `naps-track`, `naps-plot` or `naps-interactions` flag was removed or renamed.
- Requirements: the old SLEAP-era pins (`python-rapidjson`, `imgstore`, `nixio`, `qimage2ndarray`,
  `segmentation-models`, `pynwb`, `tqdm`) and `pytest`/`pytest-cov` are no longer runtime dependencies;
  `matplotlib`, `shapely` and `pyyaml` are now declared: `matplotlib` is imported lazily, only to resolve
  colour names, while `shapely` and `pyyaml` are imported at module level by `naps-plot` and `naps-interactions`. **Python 3.10 or newer is required** (previously `>=3.7`), which is the floor of `sleap-io`.
- The `naps_track.py`, `naps_plot.py` and `naps_interactions.py` files are no longer copied into `bin/` as
  loose scripts; use the `naps-track`, `naps-plot` and `naps-interactions` console commands (unchanged).
- Continuous integration: GitHub Actions run on Python 3.10-3.13 with uv, ruff, ty (the ty step is report-only), pytest with branch
  coverage, `uv build` and `twine check`; PyPI releases use trusted publishing (OIDC) from `v*` tags.

### Deprecated

- `--h5-path` is accepted and ignored (a `DeprecationWarning` is raised and a log line is written); `naps-track` always reads the tracks from `--slp-path`.
- The `min_sleap_score` argument of `naps.matching.Matching` is deprecated and ignored (a `DeprecationWarning` is raised); it is kept so that positional callers do not shift.

### Fixed

- `naps/__init__.py` no longer overwrites the module's own `__name__` with `"naps-track"`. That made
  `from naps import naps_track` look for a module called `naps-track.naps_track` and fail with
  `ModuleNotFoundError: No module named 'naps-track'` unless the submodule had already been imported.
  The distribution name is now available as `naps.__title__`; `naps.__name__` is `"naps"` again.
- Tag ids written to the output are plain integers: tracks are always named `ArUcoTag#<int>`. Older output could
  contain `ArUcoTag#17.0`, `ArUcoTag#None` or `ArUcoTag#nan` (float ids from a pandas index). Invalid
  assignments leave the instance without a track and are counted and logged.
- `update_labeled_frames` selects frames by `frame_idx` instead of slicing the list of labeled frames by
  position, and no longer swallows every exception when mapping a track to a tag.
- The cost matrix removed a vote from the sliding window with a different condition than the one used to
  add it (`tag != -1` versus `is not None`); both paths now share one definition.
- `tests/test_matching.py` asserted a nested-list result that only held through numpy scalar broadcasting.
- `TrackingArray.loadYAML` (and with it `--yaml-model` and `model_yaml=`) now raises a `ValueError` when the
  YAML file is empty or does not hold a mapping at the top level; before it returned `None` (or a list or
  scalar) and the failure surfaced later as a `KeyError`.
- An inline `--closed-set-map` that names a region twice (`"a:1,2;a:3,4"`) is now an error; before, the
  later entry silently replaced the earlier one.
- Closed-set elimination normalises the tags it sees before testing them against the closed set, so `21.0`,
  `"21"` or `True` (for tag 1) are treated as the integer tag and written as plain `int`; before, a float
  kept its type in the output and a numeric string was dropped as a foreign tag. An unhashable tag value is
  now counted as foreign instead of raising `TypeError`.
- `IdtrackerEvidence(id_to_tag=...)` normalises the fixed tags to plain `int` and raises `ValueError` for an
  invalid one (`None`, NaN, non-integral).
- `naps.tags.normalize_tag` no longer raises `OverflowError` for numbers too large for a float (for example
  `Fraction(10**400)`); such a value is not a tag id.
- Docstrings and `--help` texts that misdescribed the code were corrected (`--frame-start/--frame-end` of
  `naps-plot`, the exclusive end of `naps-interactions`, the evidence `prior` votes, legacy-mode quadrant
  regions, inference support counting, `Raises` sections).

### Removed

- `setup.py`, `requirements.txt`, `requirements-colab.txt`, `environment_apple_silicon.yml`, `.flake8`,
  `.isort.cfg`, `tests.sh` and `docs/environment.yml` (replaced by `pyproject.toml` and ruff).
- `install_with_sleap.py` (a workaround for SLEAP's numpy constraints that is no longer needed now that
  NAPS does not depend on SLEAP). It still exists on the `wolf-dev-fixing-deps` branch, which the Colab
  notebook downloads from.
- The legacy conda build files for Python 3.7 and `sleap=1.3.0` (`build.sh`, `bld.bat`,
  `conda_build_config.yaml`, `.conda/environment_build.yml`); `.conda/naps-track/meta.yaml` is now a
  noarch recipe consistent with `pyproject.toml`.

### Known limitations

- Elimination firing on a real four-colonies-per-frame video has not been demonstrated; the only real
  full-frame example in the repository is a single arena with about 46 bees. Thresholds (min support 30,
  purity 0.9, 400 px cap, 25-frame look-back) and the idtracker.ai weights are uncalibrated defaults.
- The first `--half-rolling-window-size` frames of a run are never assigned by the cost matrix.
- See `README.rst` ("Troubleshooting") for installation notes.

Release history before this entry: see the GitHub releases / tags (v1.0.6 ... v1.0.9).

[Unreleased]: https://github.com/kocherlab/naps/compare/v1.0.9...HEAD
