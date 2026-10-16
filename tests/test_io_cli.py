"""``naps-track`` I/O options (fast read / fast write) must not change the written tags."""

from __future__ import annotations

import pytest

from naps import naps_track

from ._cli_scene import make_args, tags_of


@pytest.fixture
def scene_args(tmp_path, colony, slp_factory, video_factory):
    return make_args(tmp_path, colony, slp_factory, video_factory)


OPTIONS = {
    "no_fast_read": ["--no-fast-read"],
    "no_fast_write": ["--no-fast-write"],
    "no_fast_either": ["--no-fast-read", "--no-fast-write"],
}


@pytest.mark.parametrize("option", sorted(OPTIONS))
def test_option_does_not_change_the_output(scene_args, tmp_path, option):
    assert naps_track.main(scene_args("base.slp")) == 0
    assert naps_track.main(scene_args("variant.slp", *OPTIONS[option])) == 0
    assert tags_of(tmp_path / "variant.slp") == tags_of(tmp_path / "base.slp")
