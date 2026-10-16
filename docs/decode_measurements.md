# Decode stage: the measurements behind the presets

The numbers quoted in `naps/decode.py`, `naps/video.py` and the changelog come from two benchmark
campaigns run by the authors. Both were run on the lab's own videos only (3664 x 3664 gray HEVC,
20 fps, `DICT_5X5_100`, colonies of 4 tags); treat them as measurements on that data, not as
guarantees for other cameras, tags or crop sizes, and re-measure before relying on them elsewhere.
The raw benchmark code and logs are not part of this repository (they live in the lab's analysis
repository under `workflow/idtracker/naps_core/decode/` and `.../perimeter_rate/`).

## 1. Decoder presets (Beszel Trial2, camera0)

Data: video `20230620_Beszel_Trial2_camera0`, hours 000, 020, 060, 100 and 110, all four quadrants;
per (hour, quadrant) the two most spread-out, the two most huddled and one random 6000-frame window
plus the hard "seg" windows; 300 consecutive frames from each. **111,700 crops in 94 windows**
(+-100 px luma crops around the NAPS reference tag-node position).

Reference and metrics: the reference tag is NAPS's own windowed vote, and a crop exists only where
NAPS already had a position. "Recall" is therefore per-frame decodability on frames NAPS identified,
**not** recovery of frames NAPS lost, and the reference has its own error (0.377 % of crops decode
another colony tag at the reference position, almost all of them next to a neighbour's tag node or
in two stretches where NAPS itself assigned the wrong tag). Confidence intervals are cluster
bootstraps over the 94 windows. Timings were taken while other jobs shared the node: only ratios
mean something.

| decoder (through `TagDecoder`, 100 x 100 crops) | recall % [95 % CI] | vs legacy (pp) | ms / crop |
|---|---|---|---|
| legacy: full dictionary, BGR pixels | 91.58 [89.79, 93.16] | 0 | 1.35 |
| closed set of 4 + error correction 1.0 (1 pass) | 94.73 [93.26, 96.01] | +3.15 | 0.33 |
| 2-pass cascade (+ windows 3/33/3) | 96.47 [95.12, 97.66] | +4.89 | 0.42 |
| **3-pass cascade (`CLOSED_SET_PASSES`)** | **97.52 [96.49, 98.45]** | **+5.94** | **0.44** |

* Wrong in-set tag per crop: 0.377 % legacy, 0.388 % for the 3-pass cascade (+0.011 pp: reference
  noise, unchanged by the decoder). A 3-pass decoder restricted to four tag ids that appear nowhere
  in the video accepted one of them in 8 of 446,800 trials (0.0018 %); the 2-pass cascade 0 of
  446,800; the unrestricted relaxed full-dictionary decoder returned an out-of-set id in 7 of
  111,700 crops; legacy NAPS 0 of 111,700.
* Windowed identity (+-20-frame plurality vote, an approximation of the NAPS rolling window, not
  the cost matrix): correct 98.38 -> 99.23 %, windows without any vote 1.20 -> 0.41 %.
* The levers are error correction up to the dictionary's capacity (safe only with a closed set)
  and a finer threshold sweep; CLAHE and lower thresholds help only the low-contrast "seg" windows
  (82 -> 97 % recall there), which is why they are a late cascade pass.
* Dictionary detection (`guess_dictionary`): 12 random crops around tag nodes never sufficed, 24
  were right in 2 of 3 hours, 48 in 3 of 3.

## 2. Reading frames (same data)

* The raw luma (Y) plane is **1.3 to 1.5 times** faster than `VideoCapture.read()` (1 CPU: 31 vs
  40 ms per 3664^2 frame; 8 CPUs: 8.3 to 8.9 vs 13.0 to 13.8 ms). HEVC decoding itself dominates.
  A random seek costs 1 to 3 s, so walk forward with `grab()` and seek rarely.
* For limited-range video `read()` expands luma to 0-255; the reader measures that table on frame
  0 (about 0.2 s for a 3664^2 frame) so the old pixels can be reproduced exactly. With it the new
  decode stage returned vote-for-vote identical results to `Matching._decode_video` on 600 frames
  x 16 tracks in two videos.
* End to end (600 frames, 16 tracks/frame): the luma reader with identical votes was 7 to 16 %
  faster than the legacy loop; one closed-set pass on top was 36 to 47 % faster.

## 3. `minMarkerPerimeterRate` (12 colony-trials, both cameras)

Data: 53,640 crops from 61 windows of 12 colony-trials and both cameras (hours 0 to 112, separated
and huddled bees). For the legacy full-dictionary decoder, a single closed-set pass and the 3-pass
preset, rates 0.10 to 0.30 changed not one crop at crop half-widths 50, 64 and 100 and with x1.5 /
x2 upsampling (0 lost, 0 gained, identical vote lists, unchanged wrong-in-set and decoy
false-accept rates); 16,137 frame-by-track votes of `decode_video` on full frames were identical
too. Per crop at half-width 50, rate 0.15 versus OpenCV's default 0.03: the legacy decoder 6.8x
faster, a single closed-set pass 1.7x, the 3-pass preset about 2x (about 1.0 to 1.6x for 200 px crops or
x2 upsampling). The limit is physical: at half-width 64 a rate of 0.8 lost 235 crops (-0.43 pp)
with the legacy decoder and 423 (-0.78 pp) with one closed-set pass. Keep `rate * max(crop side)`
below about 48 px, half of the smallest tag's perimeter (96 px).
