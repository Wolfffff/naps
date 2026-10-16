# The joint identity pipeline: stages, sidecar, recommended profile, limits

A short guide to `naps-track --identity-engine joint`. The option reference is `naps-track --help`
(or the CLI page) and `README.rst` ("Joint identity engine"); this page says what the stages are,
how to read the outputs, which profile the authors use and what has, and has not, been measured.

All accuracy numbers below were measured by the authors on their own data (honeybee colonies of 4
tagged bees per arena quadrant, 3664 x 3664 px, 20 fps, `DICT_5X5_100`). **There is no independent
ground truth yet:** every reference label is NAPS's own (an agreement of the legacy engine and the
production output, or the tag decodes themselves), and a human audit of 705 items is built but not
labelled. Read "precision" and "wrong" as "disagreement with NAPS-derived labels", and treat the
numbers as measurements on that data, not guarantees for other cameras, tags or colonies.

## Pipeline stages

| Stage | What it does | Module | Switch |
|---|---|---|---|
| Pose (outside NAPS) | SLEAP `.slp` with a tag node and a body node; track ids are optional | sleap-io | any producer of an `.slp` |
| Read | decode the tag around every instance's tag node | `decode`, `aruco`, `video`, `matching`, `parallel_decode`, `precomputed` | `--decoder legacy\|fast`, `--decode-workers`, `--decodes PATH` |
| Tracklets | link instances by position, velocity and pose into short high-purity tracks; SLEAP ids are only a hint | `instances`, `tracklets` | `--tracklets built\|sleap` |
| Evidence | split by region (colony/quadrant), drop decodes of tags outside the region's closed set, pick anchors (clean unique reads), demote strays | `joint`, `identity`, `quadrants`, `evidence` | `--closed-set`, `--closed-set-map`, `--closed-set-infer` |
| Inference | exact forward-backward over the injective tag assignments of each region | `identity` | `--identity-engine legacy\|joint` |
| Consistency | remove implausible recovered labels (ghosts, duplicates, teleports), fill short gaps in the sidecar only | `consistency` | `--consistency`, `--interpolate-max-gap` |
| Abstention | a recovered label is written only if its posterior is at least `--min-confidence`; windows that fail the quality criteria are withheld or kept | `identity`, `diagnostics`, `reasons` | `--min-confidence`, `--untrusted-policy` |
| Outputs | `.slp` with tags, sidecar `.identity.npz`, `--report-json` | `joint`, `naps_track` | `--identity-sidecar`, `--report-json` |
| Self-validation | hold out decodes, re-solve, score against them | `diagnostics` | `--self-validate` (off, slow) |

Anchors (clean, unique tag reads) are evidence, not inference: the output label of an anchored
instance is the decoded tag, and the model can only flag a disagreement (`disputed`). Everything
else is inference over the closed tag set, written only above the confidence threshold.

## Sidecar schema version 2

`<output>.identity.npz`, with the integer array `schema_version` = 2 (files without it are version 1).
Per-instance arrays have shape `(frames, slots)`:

- `tag` (`-1` = none), `confidence`, `source` (names in `source_names`: none, decode, tracklet,
  elimination, appearance, interpolated), `disputed`, `flag`, `region`, `inst_index`, `frames`.
- `reason`: why an instance with a position has no tag. It is nonzero exactly where the slot holds a
  valid instance and `tag == -1`. Names are in `reason_names`; the integer codes are append-only and
  never renumbered: 1 ghost_instance, 2 duplicate_within_dedupe_px, 3 excess_instance,
  4 below_min_confidence, 5 untagged_bee, 6 blocked_by_anchor, 7 consistency_dropped,
  8 untrusted_window_withheld, 9 low_confidence_in_pile, 10 no_anchor_in_tracklet, 11 no_position,
  12 outside_regions, 13 region_no_data, 14 region_unsupported_tagset.
- `tracklet_id`: the tracklet of the instance (`-1` = none), unique across regions through the
  per-region `tracklet_offset__<region>`.
- `decode_demoted`: the instance's decode was contradicted (the same tag on two instances of a frame)
  or was a minority tag on its tracklet, and was demoted to weak evidence.
- Per region `<r>`: `tags__<r>`, `tags_qualified__<r>`, `status__<r>`, `tracklet_offset__<r>`, and for
  solved regions `tag_xy__<r>` (including interpolated points), `tag_source__<r>`,
  `tag_confidence__<r>`, `tag_slot__<r>`; also `session` and `meta` (JSON).

Downstream rules: filter on `source` and `confidence`, not on the bare tag; exclude `interpolated`
points from kinematics; tag ids repeat across colonies and trials, so key on the session-qualified
name (`<session>:ArUcoTag#<int>`). `--report-json` carries the same bookkeeping per region and
window (`abstentions_by_reason`, `reason_names`, `demoted_decodes`, `n_tracklets`,
`sidecar_schema_version`, completeness before/after, jump statistics, window status).

## Recommended profile

```bash
naps-track --slp-path X.slp --video-path V.mp4 --output-path X.naps.slp \
  --identity-engine joint --closed-set-infer \
  --tracklets built --decoder fast --colony-size 4 \
  --min-confidence 0.99 --untrusted-policy withhold \
  --report-json X.report.json
```

(`--untrusted-policy` is shown at its default; `emit` is the other choice, see below.) Add the ArUco options your videos need (the authors' production values were
`--aruco-error-correction-rate 0.6 --aruco-adaptive-thresh-constant 7
--aruco-adaptive-thresh-win-size-min 3 --aruco-adaptive-thresh-win-size-max 23
--aruco-adaptive-thresh-win-size-step 10 --aruco-crop-size 50`; these are not the CLI defaults).

- **Engine:** `joint` is opt-in; the default stays `legacy`. On the authors' 12-window study the
  joint engine was not worse on any pooled metric, and its completeness was higher; its held-out
  net gain over legacy was not resolved at that sample size.
- **Closed set:** `--closed-set-infer` infers each quadrant's tag set from the video's own decodes
  (ambiguous regions abstain); `--closed-set-map` lists them explicitly.
- **`--min-confidence`:** the default stays **0.99**. On 38 held-out windows (396,323 bee-frames,
  NAPS-referenced truth) the engine confidence was well calibrated (ECE 0.0028; 0.0042 on the 24
  pre-registered windows), so a lower value trades precision for coverage. **0.98** passed the
  pre-registered adoption rule under `--untrusted-policy emit` (net, i.e. correct minus wrong, per
  bee-frame: +1.43 percentage points [+0.50, +2.53]; official wrong +0.21 per 1000 [+0.01, +0.54]).
  0.95 and 0.9 gave +2.33 and +3.02 points at +0.99 and +1.89 wrong per 1000; the authors would choose
  them only after the audit of the "misplaced" rows below. An earlier, different test (4 windows,
  leave-2 hold-out) had found a poorly calibrated [0.90, 0.95) band; the 38-window test did not
  reproduce it, but it cannot see that test's failure mode, and per-window calibration is worse in
  some pile-heavy windows (per-window ECE up to 0.099).
- **`--untrusted-policy`:** `withhold` (the default) writes only decoded labels in windows that fail
  the quality criteria; `emit` keeps the recovered labels there too. The window test is a knife-edge:
  in one window (`m10_UT2c1`) tiny parameter changes flipped the whole window to withheld, losing
  3,545 recovered labels (completeness down to 15 points), so several parameter comparisons were
  "mixed" under `withhold` and "adopt" under `emit` for that reason alone. Choose the policy
  deliberately and report it. `emit` is the setting the 0.98 result above was obtained with. In the earlier 12-window study (3-pass decoder arm) `emit` vs `withhold` gave net 89.3 vs 84.2 with 25.1 vs 19.2 wrong per 1000, i.e. more labels and more wrong ones; the
  evidence for `emit` being worse in one window is anecdotal.

## Limits found so far

- **No independent ground truth.** All references are NAPS-derived and the audit is unlabelled, so
  absolute identity accuracy is unknown, and a failure shared by both engines would be hidden.
- **ArUco is never fully replaced.** Tag reads remain the identity source. Every other evidence source
  that was tried was dropped or restricted: idtracker.ai trajectories only vote for tracks without a
  decode, a learned tag reader was inconclusive as an in-engine source, and video-object-segmentation
  trackers (Cutie, SAM2.1, DAM4SAM, CoTracker3) never reached the confidence needed to be a label
  source; the tag-blind joint engine beat all of them on continuity across tag-bracketed gaps.
- **At most 6 tags per region** (`MAX_TAGS`); a region with more tags is `unsupported_tagset`. Three
  tags plus one untagged bee is supported but was tested on synthetic data only.
- **The position node is the tag node by default.** `--position-node-name` defaults to `Thorax`,
  which the authors' skeleton does not contain (it has `thorax_a`), so the body
  position silently falls back to the tag node. The 25 px duplicate rule then merges two bodies whose
  tag nodes coincide: 90% of 2,795 wrong held-out rows were "misplaced" (the tag sits on another
  instance than the one nearest the truth bee; only 0.18 per 1000 were swaps onto another bee). This
  is an upper bound for same-bee second poses and still needs a human audit. Using `thorax_a` as
  the position node is the largest untested lever: per-window net swung from -33 to +29 points and the
  pooled net was unresolved (-1.18 [-6.87, +4.03]), because the thresholds were calibrated for tag-node
  geometry. It is not enabled by default.
- **Huddles.** Gains and losses concentrate in a few pile-heavy windows; two windows look worse than
  legacy (`m30_UT4c1`, `m10_UT2c1`) and are unadjudicated.
- **Interpolated points** (sidecar only) are within 40 px of the truth between 35 and 91% of the time
  depending on the protocol; exclude them from kinematics.
- **Parameters were tuned on the evaluation pool** (tracklet and consistency parameters, window
  thresholds, the 0.99 operating point), with no nested validation.

The decode-stage benchmark numbers are in `docs/decode_measurements.md`.
