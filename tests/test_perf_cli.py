"""``naps-track`` decode options (parallel workers, keyframe alignment) must not change the tags."""

from __future__ import annotations

import pytest

from naps import naps_track

from ._cli_scene import make_args, tags_of


@pytest.fixture
def scene_args(tmp_path, colony, slp_factory, video_factory):
    return make_args(tmp_path, colony, slp_factory, video_factory)


OPTIONS = {
    "workers": ["--decode-workers", "2", "--decode-chunk-frames", "6"],
    "workers_many_chunks": ["--decode-workers", "3", "--decode-chunk-frames", "2"],
    "workers_gop": ["--decode-workers", "2", "--decode-chunk-frames", "6", "--decode-gop", "5"],
    "chunks_only": ["--decode-chunk-frames", "4"],
    "all_cpus": ["--decode-workers", "0"],
}


@pytest.mark.parametrize("option", sorted(OPTIONS))
def test_option_does_not_change_the_output(scene_args, tmp_path, option):
    assert naps_track.main(scene_args("base.slp")) == 0
    assert naps_track.main(scene_args("variant.slp", *OPTIONS[option])) == 0
    assert tags_of(tmp_path / "variant.slp") == tags_of(tmp_path / "base.slp")


@pytest.mark.parametrize("decoder", ["legacy", "fast"])
def test_the_joint_engine_decodes_in_parallel_with_the_same_result(scene_args, tmp_path, decoder):
    joint = ["--identity-engine", "joint", "--closed-set", "5", "6", "7", "--colony-size", "3"]
    joint += ["--decoder", decoder, "--session", "S"]
    assert naps_track.main(scene_args("jbase.slp", *joint)) == 0
    parallel = ["--decode-workers", "2", "--decode-chunk-frames", "6"]
    assert naps_track.main(scene_args("jpar.slp", *joint, *parallel)) == 0
    assert tags_of(tmp_path / "jpar.slp") == tags_of(tmp_path / "jbase.slp")


def test_parallel_decode_with_the_labels_loaded_up_front(scene_args, tmp_path):
    assert naps_track.main(scene_args("base.slp")) == 0
    slow_parallel = scene_args("slow2.slp", "--no-fast-read", "--decode-workers", "2")
    assert naps_track.main(slow_parallel) == 0
    assert tags_of(tmp_path / "slow2.slp") == tags_of(tmp_path / "base.slp")


@pytest.mark.parametrize(
    "option",
    [
        ["--decode-workers", "-1"],
        ["--decode-gop", "-5"],
        ["--decode-gop-offset", "-1"],
        ["--decode-chunk-frames", "0"],
    ],
)
def test_invalid_decode_options_are_rejected(scene_args, option):
    with pytest.raises(SystemExit):
        naps_track.main(scene_args("bad.slp", *option))
