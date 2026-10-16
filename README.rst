|Stable version| |Latest version| |Documentation| |github ci| |conda| |Conda Upload| |PyPI Upload| |LICENSE|

.. |Stable version| image:: https://img.shields.io/github/v/release/kocherlab/naps?label=stable
   :target: https://github.com/kocherlab/naps/releases/
   :alt: Stable version

.. |Latest version| image:: https://img.shields.io/github/v/release/kocherlab/naps?include_prereleases&label=latest
   :target: https://github.com/kocherlab/naps/releases/
   :alt: Latest version

.. |Documentation| image::
   https://readthedocs.org/projects/naps/badge/?version=latest
   :target: https://naps.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |github ci| image::
   https://github.com/kocherlab/naps/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/kocherlab/naps/actions/workflows/ci.yml
   :alt: Continuous integration status

.. |conda| image::
   https://anaconda.org/kocherlab/naps-track/badges/version.svg
   :target: https://anaconda.org/kocherlab/naps-track

.. |Conda Upload| image::
   https://github.com/kocherlab/naps/actions/workflows/upload_conda.yml/badge.svg
   :target: https://github.com/kocherlab/naps/actions/workflows/upload_conda.yml

.. |PyPI Upload| image::
   https://github.com/kocherlab/naps/actions/workflows/python-publish.yml/badge.svg
   :target: https://github.com/kocherlab/naps/actions/workflows/python-publish.yml

.. |LICENSE| image::
   https://anaconda.org/kocherlab/naps-track/badges/license.svg
   :target: https://github.com/kocherlab/naps/blob/main/LICENSE.md

*******************************
NAPS (NAPS is ArUco Plus SLEAP)
*******************************

NAPS is a tool for researchers with two goals: (1) to quantify animal behavior over a long timescale and high resolution, with minimal human bias, and (2) to track the behavior of individuals with a high level of identity-persistence. This could be of use to researchers studying social network analysis, animal communication, task specialization, or gene-by-environment interactions. By combining deep-learning based pose estimation software with easily read and minimally invasive fiducial markers ("tags"), we provide an easy-to-use solution for producing high-quality, high-dimensional behavioral data.

.. figure:: https://raw.githubusercontent.com/kocherlab/naps/main/docs/_static/example_tracking.gif
   :width: 600px
   :align: center
   :alt: Example usage of NAPS to track a colony of common eastern bumblebees.

NAPS takes the pose-estimation tracks produced by `SLEAP <https://sleap.ai/>`_ (which are accurate within a track but whose identities are arbitrary and can swap), decodes the `ArUco <https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html>`_ tag on each animal in the video, and assigns every track the tag it is most consistently matched with. The result is a SLEAP analysis file in which tracks are named ``ArUcoTag#<id>``.

Documentation: `naps.rtfd.io <https://naps.rtfd.io/>`_. Changes between versions: see `CHANGELOG.md <https://github.com/kocherlab/naps/blob/main/CHANGELOG.md>`_.

========
Features
========
* Hybrid tracking: SLEAP pose tracks plus ArUco tag identities, matched with a rolling-window Hungarian assignment
* Built on `OpenCV <https://opencv.org/>`_ (``cv2.aruco``) and `sleap-io <https://github.com/talmolab/sleap-io>`_ (the full SLEAP/TensorFlow install is *not* needed to run NAPS)
* Opt-in closed-set identification (a known set of tags per colony or per quadrant), `idtracker.ai <https://idtracker.ai/>`_ continuity evidence, and per-video diagnostics
* Modular, extensible Python API (``naps.matching.Matching``, ``naps.evidence.EvidenceSource``) that allows drop in of different methods for marker identification

============
Installation
============
NAPS needs **Python 3.10 or newer** (the floor set by ``sleap-io``).

``pip`` / ``uv`` **(any OS)**:

.. code-block:: bash

    pip install naps-track
    # or, in a fresh environment:
    uv venv && source .venv/bin/activate && uv pip install naps-track

``conda`` **(Linux / macOS / Windows)**:

.. code-block:: bash

    conda create -n naps -c kocherlab -c conda-forge naps-track

From source (development install with test, lint, type-check and packaging tools):

.. code-block:: bash

    git clone https://github.com/kocherlab/naps.git && cd naps
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev]"

Optional extras: ``dev`` (pytest, pytest-cov, ruff, ty, pre-commit, build, twine, validate-pyproject) and ``docs`` (Sphinx). There is deliberately no extra for SLEAP: NAPS never imports it, and ``sleap`` installs ``opencv-python``, which conflicts with the OpenCV build NAPS needs (see Troubleshooting).

To produce the SLEAP predictions NAPS consumes you still need `SLEAP <https://sleap.ai/>`_ itself; install it separately following its own instructions (we recommend Google Colab, see the `basic workflow notebook <https://colab.research.google.com/github/kocherlab/naps/blob/main/docs/notebooks/naps_basic_workflow.ipynb>`_). Note that this notebook installs a development branch of NAPS (``wolf-dev-fixing-deps``) from GitHub, not the released ``naps-track`` package.

==========
Quickstart
==========
.. code-block:: bash

    naps-track \
        --slp-path example.slp \
        --video-path example.mp4 \
        --tag-node-name tag \
        --start-frame 0 --end-frame 1200 \
        --aruco-marker-set DICT_5X5_50 \
        --output-path example-naps.analysis.h5

``--slp-path`` is the SLEAP predictions file (``.slp`` with tracks), ``--video-path`` the video it was made from, ``--tag-node-name`` the skeleton node that sits on the tag and ``--aruco-marker-set`` the OpenCV dictionary of your printed tags. The ``--h5-path`` option is deprecated and ignored: NAPS always reads the tracks from ``--slp-path``. Nearly every real data set needs its ArUco detection parameters tuned (see below); the example data used in the tests lives in ``tests/data/``.

==================
Command-line tools
==================
``naps-track``
--------------
Required: ``--slp-path``, ``--video-path``, ``--start-frame``, ``--end-frame`` (zero-based, inclusive) and ``--aruco-marker-set``. Run ``naps-track --help`` for every option and default.

============================================  ===========================================================================
Option                                        Purpose
============================================  ===========================================================================
``--tag-node-name`` (``tag``)                 SLEAP node located on the ArUco tag
``--h5-path``                                 Deprecated and ignored: NAPS always reads the tracks from ``--slp-path``
``--output-path``                             Output file (``naps_output.analysis.h5``)
``--half-rolling-window-size`` (5)            Flanking frames per side in the matching window (window = 2N+1)
``--aruco-crop-size`` (100)                   Pixels around the tag node searched for a marker
``--aruco-adaptive-thresh-*``                 OpenCV adaptive-threshold parameters (win-size min/max/step, constant)
``--aruco-error-correction-rate``             OpenCV error correction rate (1 = full dictionary capability)
``--aruco-perspective-rm-ignored-margin``     OpenCV ``perspectiveRemoveIgnoredMarginPerCell``
``--aruco-min-marker-perimeter-rate``          OpenCV ``minMarkerPerimeterRate`` (detector speed, see below)
============================================  ===========================================================================

Opt-in identity options (off by default; without them NAPS behaves exactly as before):

* ``--aruco-marker-subset ID [ID ...]``: restrict the decoder to these marker ids.
* ``--aruco-min-marker-perimeter-rate RATE``: OpenCV's ``minMarkerPerimeterRate``, the shortest candidate contour the detector examines, as a fraction of the larger side of the crop (``RATE x crop side`` pixels of perimeter). OpenCV's default 0.03 is 3 px on a 100 px crop, so many tiny noise quads are tested one by one. ``0.15`` gives **identical decodes** (53,640 real crops from 12 colony-trials and both cameras, hours 0-112, separated and huddled bees, legacy full-dictionary decoder, closed-set decoder and 3-pass preset, crop half-widths 50/64/100 and x1.5/x2 upsampling: 0 crops lost or gained, every vote list identical; 16,137 frame-by-track votes of the real decode loop identical) with the legacy decoder 6.8x faster on the default-size 100 px crop (single-pass closed set 1.7x, 3-pass preset about 2x; only 1.0-1.6x on 200 px crops or x2 upsampling). The limit is the tag size: tags are 24-41 px per side (96-164 px of perimeter) and recall falls only when ``RATE x crop side`` exceeds that (0.8 on a 128 px crop, 0.6 on a 200 px crop), so keep ``RATE x crop side`` below about 48 px (0.15 is safe up to 320 px crops; the ``naps-track`` crop is ``2 x --aruco-crop-size``). Default: unchanged (OpenCV's 0.03) with the legacy decoder; ``--decoder fast`` and ``naps.decode.TagDecoder.for_closed_set`` use 0.15 (pass ``0.03`` / ``min_marker_perimeter_rate=None`` to switch off).
* ``--decodes PATH``: precomputed tag decodes (an ``.npz``; layout in the docstring of ``naps.precomputed``) for ``--slp-path``, e.g. written by a GPU pipeline that decoded the video once for SLEAP and for the tags. NAPS then never opens the video (``--video-path`` still names it and the session) and the ``--aruco-*`` decoding options are not used; works with both identity engines and is mutually exclusive with ``--decoder fast``. Instance ``j`` of a frame in the decodes file is ``LabeledFrame.instances[j]`` of the ``.slp``; NAPS exits with an error if the file does not cover ``--start-frame``..``--end-frame``, the number of instances of a frame differs from the ``.slp``, or a tag node differs from the one the decodes were computed at. Python API: ``naps.precomputed.PrecomputedDecodeSource`` (a ``DecodeSource``; ``Matching(decode_source=...)``).

Speed options (none changes the result; the tags and every dataset of the output are identical to the serial run; differential tests plus a dataset-level comparison on real 2000-frame windows):

* ``--decode-workers N`` (default 1; 0 = all CPUs of the process) decodes the video in ``N`` parallel workers over contiguous frame ranges (``--decode-chunk-frames`` sets the chunk length; ``--decode-gop 250`` starts the chunks near keyframes, ``--decode-gop-offset`` frames after one, which makes the seek to a chunk cheap). Each worker opens its video with one decoder thread (OpenCV would otherwise start one per CPU in every worker). Works with both identity engines and both ``--decoder`` choices; not used with ``--decodes``. Only the video decode runs in parallel: the rolling-window vote accumulation stays serial because its tie-breaking depends on the whole job. With parallel workers the SLEAP file (needed again to write the output) is loaded in a background thread while the video is decoded.
* ``--no-fast-read`` and ``--no-fast-write`` fall back to ``sleap-io`` for reading the tag-node positions from the ``.slp`` and for writing the output (by default the positions are read straight from the file's tables and the output is written with a vectorised writer that is verified against ``sleap-io``'s own for the supported versions and used only then). The cost matrix keeps its rolling window as a small dictionary and array instead of a ``pandas`` frame per video frame (``naps.rolling_votes``), with the same row and column order, so ``scipy``'s tie-breaking, and therefore the assignment, is unchanged.
* ``--closed-set ID [ID ...]``: the N tags expected in the video (e.g. one quadrant). The input must contain only those animals (e.g. a per-quadrant crop of the SLEAP file and video). Votes for any other tag are dropped and closed-set elimination runs: in a frame where exactly one expected tag and exactly one tracked instance are unassigned, the instance gets that tag, provided it lies within ``--closed-set-max-jump`` pixels (per frame of gap) of where that tag was last or next seen; otherwise NAPS abstains. ``--no-closed-set-elimination`` keeps only the vote filter, ``--closed-set-max-distance`` caps the continuity threshold in absolute pixels.
* ``--closed-set-map SPEC``: for a full frame holding several independent colonies (e.g. four 4-bee colonies in the quadrants of one frame), explicit tag sets per region, either inline (``"top_left:45,46,47,48;top_right:49,50,51,52;..."``) or a ``.json``/``.yaml`` file (``{region: [tags]}`` or ``{tags: {...}, rois: {region: [x0, y0, x1, y1]}}``). Each tracked instance is assigned to a region by the position of its tag node (or ``--position-node-name``, default ``Thorax``, when the tag node is NaN); votes for tags that do not belong to that region are dropped and elimination runs per region. Regions not listed keep the legacy behaviour.
* ``--closed-set-infer``: infer the tag sets from the video's own ArUco decodes instead of listing them. A tag belongs to a region if it was decoded there in at least ``--closed-set-min-support`` frames (and ``--closed-set-min-fraction`` of the region's frames) and at least ``--closed-set-min-purity`` of its decode frames are in that region; a tag well supported in two regions is refused and both regions abstain. A region normally gets exactly ``--closed-set-size`` (default 4) tags. A region with more candidate tags is also accepted if its top ``--closed-set-size`` tags clearly dominate (the weakest of them has at least 3 times the support of the next tag; this ratio is internal, not a command-line option), and then only those tags are used.
* Regions for the two options above: ``--quadrant-grid ROWS COLS`` (default ``2 2``: ``top_left``, ``top_right``, ``bottom_left``, ``bottom_right``) or explicit, repeatable ``--quadrant-roi NAME X0 Y0 X1 Y1``. ``--closed-set`` is mutually exclusive with ``--closed-set-map`` and ``--closed-set-infer``.
* ``--idtracker-trajectories PATH``: an idtracker.ai v6 ``trajectories.npy`` used only as continuity evidence (ArUco stays the identity source and idtracker.ai only votes for tracks with no ArUco decode in that frame). Requires ``--closed-set`` or one of the quadrant modes. Tune it with ``--idtracker-offset X Y`` (crop origin), ``--idtracker-frame-offset``, ``--idtracker-weight``, ``--idtracker-min-prob``, ``--idtracker-max-dist``, ``--idtracker-min-margin``, ``--idtracker-min-support`` and ``--idtracker-min-purity``. idtracker.ai labels are arbitrary, so each animal is aligned to a tag by co-occurrence with the ArUco decodes and ignored unless it has enough unambiguous co-occurrences.
* ``--strict-carry-forward``: only carry the previous frame's tag into tracks the assignment left unmatched.
* ``--report-json PATH`` (works in every mode): per-video diagnostics: frames with 4/3/2/1/0 tags per region before and after elimination, instances left without a tag by cause, eliminated assignments, abstention reasons, inferred tag sets and wrong-region votes.

Tag ids are plain integers everywhere and tracks are always named ``ArUcoTag#<int>``; instances without a valid tag get no track.

Logging: ``-v``/``--verbose`` also logs debugging messages; ``-q``/``--quiet`` logs only warnings and errors (default: INFO).

Joint identity engine (opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``--identity-engine joint`` (default ``legacy``) replaces the rolling-window assignment by one exact probabilistic inference over the whole video. It needs the closed set of tags (``--closed-set``, ``--closed-set-map`` or ``--closed-set-infer``); under this engine the region size is ``--colony-size`` (default 4) and ``--closed-set-size`` is ignored. What it does differently:

* It **builds its own tracklets** from the SLEAP instances (position, velocity, pose; competing-instance breaks at near-passes), so it works on an untracked ``.slp``; ``--tracklets sleap`` uses the SLEAP track ids as the tracklets but splits them at implausible jumps, ``--tracklets built`` (default) only uses them as a hint.
* It **decodes the tag around every instance** (tracked or not; ``--decoder fast`` uses the faster decode stage) and treats every decode as evidence for an injective assignment of the colony's tags to the animals present: two animals never share a tag, and when all other tags are taken the last animal must carry the remaining one. Clean, unique decodes are immutable anchors (the model may flag a *disputed* decode, never changes it); isolated misreads (a tag that is a tiny minority on a tracklet, or decoded on two animals at once) are demoted to weak evidence so they cannot propagate. A demoted decode is not protected: that instance is treated as if it had no decode and may be labelled with another tag. The demotions are counted in the report (``demoted_decodes``: conflicts, strays, how many were relabelled or confirmed) and marked per instance in the sidecar (``decode_demoted``).
* It labels **every frame, including the first and last**, never gives a tag that was not supported, and abstains (the instance stays untracked) unless the posterior is at least ``--min-confidence`` (default 0.99; see ``docs/identity_pipeline.md`` for what was measured at lower values).
* A **temporal-consistency stage** (``--consistency``, default on; ``--no-consistency`` switches all of it off, including the gap interpolation) drops recovered labels that teleport, sit on another tag, or sit on a duplicate (ghost) instance, and fills gaps of up to ``--interpolate-max-gap`` frames only where plausible (sidecar only; no instances are invented). The thresholds are calibrated defaults of ``naps.consistency.ConsistencyParams``.
* Guard cases: a region of three tags is solved as three tags plus one untagged bee (``--colony-size``, default 4); a region without decodes is ``no_data``, one with another number of tags ``unsupported_tagset``; neither is guessed.
* Tag ids repeat across colonies and trials, so the sidecar and the ``identity`` of every labelled instance carry the session-qualified name ``<session>:ArUcoTag#<int>`` (``--session``, default the directory of the video); track names stay ``ArUcoTag#<int>``.

Outputs: the re-tracked ``--output-path`` (labelled instances get the track ``ArUcoTag#<int>``, ``tracking_score`` and ``identity_score`` = posterior; abstained instances stay untracked), ``--identity-sidecar`` (``.npz``, schema version 2 below, with per-frame confidence, source code, dispute and abstention flag for every instance, a ``reason`` code for every instance left without a tag, a ``tracklet_id`` for every instance, and the per-tag positions incl. interpolated ones) and ``--report-json`` (per region and window of ``--window-frames`` frames, default 6000: completeness before/after, sources of every label, abstentions by cause and by reason, demoted decodes, disputes, jump and duplicate statistics, window status ``trusted``/``uncertain``/``unvalidated``/``no_repair_needed``; regions that cannot be solved are ``no_data`` or ``unsupported_tagset`` and have no windows; ``--self-validate`` adds a hold-out check: the decodes of each tag are removed, the model re-run, and its precision/coverage reported). In windows that fail the quality criteria recovered labels are withheld (``--untrusted-policy emit`` keeps them). Python API: ``naps.joint.JointMatching``, ``naps.identity.solve_region``.

Identity sidecar, schema version 2 (``schema_version`` array; files without it are version 1, which has none of the arrays marked *new*). Per-instance arrays have shape ``(frames, slots)`` and are indexed like the instance table of the run (``inst_index`` gives the position of the instance in its labeled frame): ``tag`` (tag id, -1 = untracked), ``confidence``, ``source`` (``source_names``), ``disputed``, ``flag`` (``flag_names``), ``region`` (index into ``region_names``), ``reason`` (*new*), ``tracklet_id`` (*new*) and ``decode_demoted`` (*new*). ``tracklet_id`` is -1 for an instance in no tracklet (duplicate, culled for exceeding the colony size, no position, unsolved region) and unique across the regions of the file (``tracklet_offset__<region>`` is added to the region's own ids, -1 for an unsolved region); ghost instances stay in their tracklet. ``reason`` is nonzero exactly where an instance with a position has ``tag == -1``; the names are in ``reason_names`` and the codes are stable (new reasons are appended, never renumbered):

* ``ghost_instance`` (1): a duplicate detection of a decoded neighbour (also when it was culled for excess); ``duplicate_within_dedupe_px`` (2): the same bee detected twice (closer than 25 px to a higher-ranked instance); ``excess_instance`` (3): more instances than bees, the lowest-ranked is culled;
* ``below_min_confidence`` (4): the posterior is under ``--min-confidence``; ``low_confidence_in_pile`` (9): the same, with another bee within 40 px; ``no_anchor_in_tracklet`` (10): the same, and no decode was accepted anywhere on the instance's tracklet;
* ``untagged_bee`` (5): the model assigns the instance to a bee without a tag; ``blocked_by_anchor`` (6): the model's tag is already decoded on another instance of the frame;
* ``consistency_dropped`` (7): a recovered label removed by the consistency stage; ``untrusted_window_withheld`` (8): a recovered label withheld in a window that failed the quality criteria;
* ``no_position`` (11): the SLEAP instance has no finite node; ``outside_regions`` (12): in no region (quadrant modes); ``region_no_data`` (13) and ``region_unsupported_tagset`` (14): the region was not solved.

The ``--report-json`` repeats the counts as ``abstentions_by_reason`` (overall and per region) with ``reason_names`` and ``sidecar_schema_version``.
``naps-plot`` and ``naps-interactions``
---------------------------------------
``naps-plot`` renders model overlays on a video (``--sleap-file``, ``--yaml-model``, ``--interactions-file``, ``--video-file``, ``--models-to-plot`` (required), ``--out-file``; optional ``--limit-interactions``, ``--frame-start``/``--frame-end``) and ``naps-interactions`` assigns interactions from a tracking h5 (``--h5-file``, ``--yaml-model``, ``--out-file``). Both take ``--help``.

Python API
----------
.. code-block:: python

    from naps.matching import Matching
    from naps.evidence import EvidenceSource  # implement this to add your own identity evidence
    # Matching(..., evidence_sources=[...], closed_set=[...])  # or quadrants=QuadrantScope(...): mutually exclusive with closed_set

===============
Troubleshooting
===============
* **cv2.aruco is missing or broken**: install exactly one OpenCV distribution. NAPS depends on ``opencv-contrib-python-headless``; having ``opencv-python`` (for example pulled in by ``sleap-io[opencv]``) next to it overwrites ``cv2`` and breaks ``aruco``. Run ``pip uninstall opencv-python opencv-python-headless`` and then ``pip install --force-reinstall opencv-contrib-python-headless``.
* **No tags are decoded / identities flip**: ArUco detection is data set specific. Increase ``--aruco-crop-size`` until the whole marker (but not its neighbours) fits, widen ``--aruco-adaptive-thresh-win-size-min/max``, and try a smaller ``--aruco-error-correction-rate`` if you see false positives. A larger ``--half-rolling-window-size`` gives more robust tracks but slower recovery after an identity swap.
* **numpy version conflicts with SLEAP**: NAPS itself does not import SLEAP or TensorFlow. Run SLEAP (training and inference) in its own environment and NAPS in another; only the ``.slp``/``.h5`` file crosses between them.
* **The first frames have no tag**: the first ``--half-rolling-window-size`` frames of a run are never assigned by the cost matrix; start a few frames before the frames you need.
* **A closed-set run warns that elimination will abstain**: ``--closed-set`` assumes the input holds only the listed animals. For a full frame with several colonies use ``--closed-set-map`` or ``--closed-set-infer``.
* **Python 3.7-3.9**: no longer supported (``sleap-io`` needs 3.10). Use NAPS 1.0.x on older interpreters.

===========
Development
===========
.. code-block:: bash

    uv pip install -e ".[dev]"
    pytest                                  # tests (add --cov for branch coverage)
    ruff check . && ruff format --check .   # lint and formatting
    ty check                                # type checking
    pre-commit install                      # run the above on every commit
    python -m build && twine check dist/*   # build the sdist and wheel

==========
References
==========

Manuscript: `https://doi.org/10.1101/2022.12.07.518416 <https://doi.org/10.1101/2022.12.07.518416>`_

If you use NAPS in your research, please cite:

BibTeX
------
.. code-block:: bibtex

   @article{wolf2022naps,
      title={NAPS: Integrating pose estimation and tag-based tracking},
      author={Wolf, Scott W and Ruttenberg, Dee M and Knapp, Daniel Y and Webb, Andrew E and Traniello, Ian M and McKenzie-Smith, Grace C and Leheny, Sophie A and Shaevitz, Joshua W and Kocher, Sarah D},
      journal={bioRxiv},
      year={2022},
      publisher={Cold Spring Harbor Laboratory}
   }


======
Issues
======

Issues with NAPS?
-----------------

1. Check the `docs <https://naps.rtfd.io/>`_ and the Troubleshooting section above.
2. Search the `issues on GitHub <https://github.com/kocherlab/naps/issues>`_ or open a new one.

============
Contributors
============

* **Scott Wolf**, Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Dee Ruttenberg**, Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Daniel Knapp**, Department of Physics, Princeton University
* **Andrew Webb**, Department of Ecology and Evolutionary Biology and Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Ian Traniello**, Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Grace McKenzie-Smith**, Department of Physics and Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Sophie Leheny**, Department of Molecular Biology, Princeton University
* **Joshua Shaevitz**, Department of Physics and Lewis-Sigler Institute for Integrative Genomics, Princeton University
* **Sarah Kocher**, Department of Ecology and Evolutionary Biology and Lewis-Sigler Institute for Integrative Genomics, Princeton University

NAPS was created in the `Kocher <https://kocherlab.princeton.edu/>`_  and `Shaevitz <https://shaevitzlab.princeton.edu/>`_ labs at Princeton University.

=======
License
=======

NAPS is licensed under the MIT license. See the `LICENSE <https://github.com/kocherlab/naps/blob/main/LICENSE.md>`_ file for details.

================
Acknowledgements
================

Much of the structure and content of the README and the documentation is borrowed from the `SLEAP repository <https://github.com/talmolab/sleap>`_.
