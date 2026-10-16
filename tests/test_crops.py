"""Tests of the batched crop extractor (``naps.crops``) and its torch twin (``naps.crops_torch``).

Everything is seeded; the torch tests are skipped when torch is missing and the CUDA ones when no
GPU is visible.
"""

import math

import cv2
import numpy as np
import pytest

from naps.crops import (
    CropBatch,
    CropExtractor,
    heading_angles,
    resample_windows,
    stretch_crops,
)

SEEDS = range(8)


def naive_crop(frame, xy, half):
    """The legacy rule written out per crop (negative stops clamped to an empty crop)."""
    if np.isnan(xy).any() or np.isinf(xy).any():
        return None
    height, width = frame.shape
    y0, y1 = max(int(xy[1]) - half, 0), max(min(int(xy[1]) + half, height - 1), 0)
    x0, x1 = max(int(xy[0]) - half, 0), max(min(int(xy[0]) + half, width - 1), 0)
    return frame[y0:y1, x0:x1]


def random_frame(rng, shape=(64, 72)):
    return rng.integers(0, 256, shape, dtype=np.uint8)


def smooth_frame(rng, shape=(120, 130)):
    """A band-limited frame, so the sub-pixel resamplers agree to a grey level or two."""
    noise = rng.normal(size=shape).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 2.5)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return (smooth * 255).astype(np.uint8)


def random_centres(rng, n, shape, margin=30):
    height, width = shape
    xy = np.column_stack(
        [rng.uniform(-margin, width + margin, n), rng.uniform(-margin, height + margin, n)]
    )
    xy[rng.random(n) < 0.15, 0] = np.nan
    xy[rng.random(n) < 0.05, 1] = np.inf
    xy[rng.random(n) < 0.05, 0] = -np.inf
    return xy


# --------------------------------------------------------------------------- gather mode
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("half", [4, 9])
def test_view_equals_naive_slicing(seed, half):
    rng = np.random.default_rng(seed)
    frame = random_frame(rng)
    xy = random_centres(rng, 60, frame.shape, margin=3 * half)
    batch = CropExtractor(half).extract(frame, xy)
    assert isinstance(batch, CropBatch)
    assert batch.crops.shape == (60, 2 * half, 2 * half)
    for i, centre in enumerate(xy):
        want = naive_crop(frame, centre, half)
        got = batch.view(i)
        if want is None:
            assert got is None
            assert not batch.valid[i]
            assert (batch.crops[i] == 0).all()
        else:
            assert got is not None
            assert got.shape == want.shape
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("seed", SEEDS)
def test_legacy_views_equal_naive_slicing(seed):
    rng = np.random.default_rng(seed)
    frame = random_frame(rng)
    xy = random_centres(rng, 40, frame.shape)
    views = CropExtractor(7).legacy_views(frame, xy)
    for centre, got in zip(xy, views, strict=True):
        want = naive_crop(frame, centre, 7)
        if want is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, want)
            assert got is not None
            assert got.base is not None  # a view, not a copy


def test_matches_decode_crop_around_in_the_supported_range():
    decode = pytest.importorskip("naps.decode")
    rng = np.random.default_rng(3)
    frame = random_frame(rng, (80, 90))
    xy = rng.uniform(-10, 100, size=(50, 2))
    batch = CropExtractor(12).extract(frame, xy)
    for i, centre in enumerate(xy):
        want = decode.crop_around(frame, centre, 12)
        np.testing.assert_array_equal(batch.view(i), want)


def test_far_below_the_low_edge_is_empty_not_the_legacy_bogus_crop():
    frame = np.arange(40 * 50, dtype=np.int64).astype(np.uint8).reshape(40, 50)
    batch = CropExtractor(5).extract(frame, [[-30.0, 10.0], [10.0, -30.0], [500.0, 10.0]])
    for i in range(3):
        view = batch.view(i)
        assert view is not None
        assert view.size == 0
    assert not batch.nonempty.any()
    assert batch.valid.all()


def test_interior_crop_is_the_exact_window_and_origin():
    frame = random_frame(np.random.default_rng(0), (50, 60))
    batch = CropExtractor(8).extract(frame, [[30.7, 20.2]])
    np.testing.assert_array_equal(batch.crops[0], frame[12:28, 22:38])
    np.testing.assert_array_equal(batch.origins[0], [22, 12])
    np.testing.assert_array_equal(batch.boxes[0], [0, 0, 16, 16])
    np.testing.assert_array_equal(batch.frame_points(0, [[8, 8]])[0], [30, 20])
    np.testing.assert_allclose(batch.crop_points(0, [[30, 20]])[0], [8, 8])


def test_negative_fractions_truncate_toward_zero_like_int():
    frame = random_frame(np.random.default_rng(1), (30, 30))
    batch = CropExtractor(4).extract(frame, [[-0.7, 3.9]])
    np.testing.assert_array_equal(batch.origins[0], [-4, -1])
    want = naive_crop(frame, np.array([-0.7, 3.9]), 4)
    np.testing.assert_array_equal(batch.view(0), want)


def test_constant_padding_value_and_boxes():
    frame = np.full((20, 20), 200, dtype=np.uint8)
    batch = CropExtractor(6, fill=7).extract(frame, [[2.0, 3.0], [19.0, 19.0]])
    crop = batch.crops[0]
    assert (crop[:3, :] == 7).all()  # rows above the image (y0 = -3)
    assert (crop[:, :4] == 7).all()  # columns left of the image (x0 = -4)
    assert (crop[3:, 4:] == 200).all()
    np.testing.assert_array_equal(batch.boxes[0], [4, 3, 12, 12])
    far = batch.crops[1]  # x0 = y0 = 13: the right/bottom part is outside, incl. the last column
    assert (far[:7, :7] == 200).all()
    assert (far[7:, :] == 7).all()
    np.testing.assert_array_equal(batch.boxes[1], [0, 0, 6, 6])


def test_legacy_clip_false_includes_the_last_row_and_column():
    frame = random_frame(np.random.default_rng(2), (20, 20))
    clip = CropExtractor(6).extract(frame, [[19.0, 19.0]])
    keep = CropExtractor(6, legacy_clip=False).extract(frame, [[19.0, 19.0]])
    np.testing.assert_array_equal(clip.boxes[0], [0, 0, 6, 6])
    np.testing.assert_array_equal(keep.boxes[0], [0, 0, 7, 7])
    np.testing.assert_array_equal(clip.crops, keep.crops)


def test_replicate_padding_repeats_the_edge_pixels():
    rng = np.random.default_rng(4)
    frame = random_frame(rng, (24, 26))
    batch = CropExtractor(6, pad="replicate").extract(
        frame, [[1.0, 2.0], [24.0, 23.0], [np.nan, 0]]
    )
    for i in (0, 1):
        x0, y0 = batch.origins[i].astype(int)
        ys = np.clip(np.arange(y0, y0 + 12), 0, 23)
        xs = np.clip(np.arange(x0, x0 + 12), 0, 25)
        np.testing.assert_array_equal(batch.crops[i], frame[np.ix_(ys, xs)])
    assert (batch.crops[2] == 0).all()  # invalid rows take the fill value


def test_invalid_centres_get_no_view_and_zero_boxes():
    frame = random_frame(np.random.default_rng(5))
    batch = CropExtractor(5, fill=9).extract(frame, [[np.nan, 4.0], [3.0, np.inf], [10.0, 10.0]])
    assert batch.valid.tolist() == [False, False, True]
    assert batch.nonempty.tolist() == [False, False, True]
    assert batch.view(0) is None
    assert batch.view(1) is None
    assert (batch.crops[:2] == 9).all()
    assert (batch.boxes[:2] == 0).all()


def test_huge_centres_do_not_overflow():
    frame = random_frame(np.random.default_rng(6), (30, 30))
    batch = CropExtractor(4).extract(frame, [[1e300, -1e300], [5.0, 5.0]])
    assert batch.valid.all()
    assert not batch.nonempty[0]
    assert batch.nonempty[1]


def test_stack_sequence_and_frame_index():
    rng = np.random.default_rng(7)
    frames = [random_frame(rng, (40, 50)), random_frame(rng, (30, 36)), random_frame(rng, (40, 50))]
    xy = np.array([[20.0, 20.0], [15.0, 12.0], [25.0, 18.0], [8.0, 9.0]])
    index = [2, 1, 0, 1]
    ex = CropExtractor(6)
    seq = ex.extract(frames, xy, frame_index=index)
    for i, (centre, k) in enumerate(zip(xy, index, strict=True)):
        np.testing.assert_array_equal(seq.view(i), naive_crop(frames[k], centre, 6))
    np.testing.assert_array_equal(seq.frame_index, index)
    stack = np.stack([frames[0], frames[2]])
    rows = np.stack([xy[:2], xy[2:]])  # (2, 2, 2)
    batch = ex.extract(stack, rows)
    np.testing.assert_array_equal(batch.frame_index, [0, 0, 1, 1])
    assert batch.per_frame(2).shape == (2, 2, 12, 12)
    np.testing.assert_array_equal(batch.view(3), naive_crop(stack[1], xy[3], 6))
    mapping = batch.to_mapping(["a", "b", "c", "d"])
    assert list(mapping) == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(mapping["c"], naive_crop(stack[1], xy[2], 6))


def test_out_buffer_is_filled_and_returned_as_crops():
    frame = random_frame(np.random.default_rng(8))
    out = np.zeros((2, 10, 10), dtype=np.uint8)
    batch = CropExtractor(5).extract(frame, [[20.0, 20.0], [30.0, 25.0]], out=out)
    assert batch.crops is out
    np.testing.assert_array_equal(out[0], frame[15:25, 15:25])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 9, 10), dtype=np.uint8),
        np.zeros((2, 10, 10), dtype=np.int16),
        np.zeros((2, 10, 20), dtype=np.uint8)[:, :, ::2],
    ],
)
def test_bad_out_buffer_is_rejected(bad):
    frame = random_frame(np.random.default_rng(8))
    with pytest.raises(ValueError, match="out must be"):
        CropExtractor(5).extract(frame, [[20.0, 20.0], [30.0, 25.0]], out=bad)


def test_batch_helpers_reject_bad_arguments():
    frame = random_frame(np.random.default_rng(9))
    batch = CropExtractor(5).extract(frame, [[20.0, 20.0], [30.0, 25.0], [4.0, 4.0]])
    with pytest.raises(ValueError, match="keys"):
        batch.to_mapping(["only-one"])
    with pytest.raises(ValueError, match="split"):
        batch.per_frame(2)
    with pytest.raises(ValueError, match="split"):
        batch.per_frame(0)
    assert len(batch) == 3


def test_empty_batch():
    frame = random_frame(np.random.default_rng(9))
    lut = np.arange(256, dtype=np.uint8)
    batch = CropExtractor(5, lut=lut, stretch=(1, 99)).extract(frame, np.empty((0, 2)))
    assert batch.crops.shape == (0, 10, 10)
    assert CropExtractor(5).legacy_views(frame, np.empty((0, 2))) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"half": 0}, "positive"),
        ({"scale": 0.0}, "positive"),
        ({"pad": "mirror"}, "pad"),
        ({"interpolation": "lanczos"}, "interpolation"),
        ({"backend": "gpu"}, "backend"),
        ({"backend": "numpy", "interpolation": "cubic"}, "cubic"),
        ({"fill": 300}, "fill"),
        ({"lut": np.zeros(10, dtype=np.uint8)}, "lut"),
        ({"lut": np.zeros(256, dtype=np.int32)}, "lut"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CropExtractor(**kwargs)


def test_input_validation():
    ex = CropExtractor(4)
    frame = random_frame(np.random.default_rng(1), (20, 20))
    with pytest.raises(ValueError, match="uint8"):
        ex.extract(frame.astype(np.float32)[None], [[5.0, 5.0]])
    with pytest.raises(ValueError, match="2-D uint8"):
        ex.extract(frame.astype(np.float32), [[5.0, 5.0]])
    with pytest.raises(ValueError, match="2-D uint8"):
        ex.extract([frame[None]], [[5.0, 5.0]])
    with pytest.raises(ValueError, match="shape"):
        ex.extract(frame, [5.0, 5.0])
    with pytest.raises(ValueError, match="shape"):
        ex.extract(frame, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="frame_index"):
        ex.extract(frame, [[5.0, 5.0]], frame_index=[1])
    with pytest.raises(ValueError, match="frame_index"):
        ex.extract(frame, [[5.0, 5.0]], frame_index=[-1])
    with pytest.raises(ValueError, match="frame_index"):
        ex.extract(frame, [[5.0, 5.0]], frame_index=[0, 0])
    with pytest.raises(ValueError, match="cannot be combined"):
        ex.extract(frame[None], np.zeros((1, 2, 2)), frame_index=[0, 0])
    with pytest.raises(ValueError, match="angles"):
        ex.extract(frame, [[5.0, 5.0], [6.0, 6.0]], angles=[0.0])


# ----------------------------------------------------------------------- lut / stretch
def test_lut_is_applied_after_the_crop():
    rng = np.random.default_rng(10)
    frame = random_frame(rng)
    lut = (255 - np.arange(256)).astype(np.uint8)
    plain = CropExtractor(5).extract(frame, [[30.0, 30.0]])
    mapped = CropExtractor(5, lut=lut).extract(frame, [[30.0, 30.0]])
    np.testing.assert_array_equal(mapped.crops, lut[plain.crops])


def test_stretch_maps_percentiles_to_the_full_range():
    rng = np.random.default_rng(11)
    crops = rng.integers(40, 120, (5, 20, 20), dtype=np.uint8)
    out = stretch_crops(crops, (1.0, 99.0))
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255
    flat = np.full((1, 6, 6), 77, dtype=np.uint8)
    assert (stretch_crops(flat, (1.0, 99.0)) == 0).all()
    ex = CropExtractor(8, stretch=(2.0, 98.0))
    frame = random_frame(rng)
    batch = ex.extract(frame, [[30.0, 30.0]])
    np.testing.assert_array_equal(
        batch.crops,
        stretch_crops(CropExtractor(8).extract(frame, [[30.0, 30.0]]).crops, (2.0, 98.0)),
    )


# ------------------------------------------------------------------------- affine mode
@pytest.mark.parametrize("backend", ["cv2", "numpy"])
@pytest.mark.parametrize("pad", ["constant", "replicate"])
@pytest.mark.parametrize("seed", SEEDS)
def test_zero_rotation_unit_scale_reproduces_the_gather_crops(backend, pad, seed):
    rng = np.random.default_rng(seed)
    frame = random_frame(rng, (60, 70))
    xy = random_centres(rng, 30, frame.shape, margin=20)
    want = CropExtractor(8, pad=pad).extract(frame, xy)
    got = CropExtractor(8, pad=pad, backend=backend).extract(frame, xy, angles=np.zeros(len(xy)))
    np.testing.assert_array_equal(got.crops, want.crops)
    np.testing.assert_array_equal(got.valid, want.valid)
    np.testing.assert_allclose(got.transforms, want.transforms, atol=1e-9)


@pytest.mark.parametrize("backend", ["cv2", "numpy"])
def test_quarter_turn_is_a_lattice_rotation(backend):
    rng = np.random.default_rng(12)
    frame = random_frame(rng, (80, 90))
    half, cx, cy = 10, 40, 35
    batch = CropExtractor(half, backend=backend).extract(frame, [[cx, cy]], angles=[math.pi / 2])
    u = np.arange(2 * half)
    # x = cx - (v - half), y = cy + (u - half)
    want = frame[cy + (u[None, :] - half), cx - (u[:, None] - half)]
    np.testing.assert_array_equal(batch.crops[0], want)


@pytest.mark.parametrize("seed", SEEDS)
def test_cv2_and_numpy_backends_agree_on_smooth_images(seed):
    rng = np.random.default_rng(seed)
    frame = smooth_frame(rng)
    xy = rng.uniform(20, 100, size=(12, 2))
    angles = rng.uniform(-math.pi, math.pi, 12)
    for scale in (1.0, 2.0, 0.75):
        a = CropExtractor(9, scale=scale, subpixel=True).extract(frame, xy, angles=angles)
        b = CropExtractor(9, scale=scale, subpixel=True, backend="numpy").extract(
            frame, xy, angles=angles
        )
        assert a.crops.shape[1] == round(18 * scale)
        diff = np.abs(a.crops.astype(int) - b.crops.astype(int))
        assert diff.max() <= 2
        assert diff.mean() < 0.4


def test_nearest_interpolation_at_integer_lattice_points_is_exact():
    rng = np.random.default_rng(13)
    frame = random_frame(rng, (50, 50))
    for backend in ("cv2", "numpy"):
        batch = CropExtractor(6, interpolation="nearest", backend=backend).extract(
            frame, [[25.0, 24.0]], angles=[0.0]
        )
        np.testing.assert_array_equal(batch.crops[0], frame[18:30, 19:31])


def test_cubic_interpolation_runs_with_cv2():
    frame = smooth_frame(np.random.default_rng(14))
    cubic = CropExtractor(8, scale=1.5, interpolation="cubic").extract(frame, [[60.3, 50.2]])
    linear = CropExtractor(8, scale=1.5).extract(frame, [[60.3, 50.2]])
    assert np.abs(cubic.crops.astype(int) - linear.crops.astype(int)).max() <= 6


def test_upsampled_crop_geometry_and_point_mapping():
    frame = smooth_frame(np.random.default_rng(15))
    batch = CropExtractor(10, scale=2.0, subpixel=True).extract(frame, [[50.25, 60.5]])
    assert batch.size == 40
    # the crop centre (S/2, S/2) is the (sub-pixel) tag node; one crop pixel is half a frame pixel
    np.testing.assert_allclose(batch.frame_points(0, [[20, 20]])[0], [50.25, 60.5])
    np.testing.assert_allclose(batch.frame_points(0, [[22, 20]])[0], [51.25, 60.5])
    round_trip = batch.crop_points(0, batch.frame_points(0, [[3.5, 7.25]]))
    np.testing.assert_allclose(round_trip, [[3.5, 7.25]])


def test_rotation_maps_the_heading_onto_the_crop_x_axis():
    angle = heading_angles([[10.0, 12.0]], [[10.0, 4.0]])  # head below the abdomen: +y
    assert angle[0] == pytest.approx(math.pi / 2)
    frame = smooth_frame(np.random.default_rng(16))
    batch = CropExtractor(9, subpixel=True).extract(frame, [[60.0, 60.0]], angles=angle)
    # one step along the crop's +x axis is one pixel along +y in the frame
    np.testing.assert_allclose(batch.frame_points(0, [[10, 9]])[0], [60.0, 61.0], atol=1e-9)


def test_nan_angle_means_no_rotation_and_invalid_centres_are_filled():
    frame = random_frame(np.random.default_rng(17), (60, 60))
    ex = CropExtractor(7, fill=5)
    plain = ex.extract(frame, [[30.0, 30.0]], angles=[0.0])
    nan_angle = ex.extract(frame, [[30.0, 30.0], [np.nan, 3.0]], angles=[np.nan, 0.0])
    np.testing.assert_array_equal(nan_angle.crops[0], plain.crops[0])
    assert (nan_angle.crops[1] == 5).all()
    assert nan_angle.valid.tolist() == [True, False]
    heading = heading_angles([[np.nan, 1.0]], [[0.0, 0.0]])
    assert np.isnan(heading[0])


def test_affine_boxes_flag_crops_that_miss_the_frame():
    frame = random_frame(np.random.default_rng(18), (50, 50))
    batch = CropExtractor(6, subpixel=True).extract(
        frame, [[25.0, 25.0], [-40.0, 25.0], [200.0, 25.0], [25.0, -40.0], [25.0, 200.0]]
    )
    assert batch.nonempty.tolist() == [True, False, False, False, False]
    assert (batch.crops[1:] == 0).all()
    np.testing.assert_array_equal(batch.boxes[0], [0, 0, 12, 12])


def test_affine_angles_shape_is_validated():
    frame = random_frame(np.random.default_rng(19), (40, 40))
    with pytest.raises(ValueError, match="angles"):
        CropExtractor(5, subpixel=True).extract(frame, [[20.0, 20.0]], angles=[0.0, 1.0])


def test_affine_over_a_stack_uses_the_right_frame():
    rng = np.random.default_rng(20)
    stack = np.stack([smooth_frame(rng, (90, 90)), smooth_frame(rng, (90, 90))])
    xy = np.array([[45.0, 45.0], [40.0, 50.0]])
    for backend in ("cv2", "numpy"):
        both = CropExtractor(8, subpixel=True, backend=backend).extract(
            stack, xy, angles=[0.3, -0.7], frame_index=[1, 0]
        )
        for i, k in enumerate((1, 0)):
            single = CropExtractor(8, subpixel=True, backend=backend).extract(
                stack[k], xy[i : i + 1], angles=[(0.3, -0.7)[i]]
            )
            np.testing.assert_array_equal(both.crops[i], single.crops[0])


def test_resample_windows_rejects_cubic_and_reads_fill_outside():
    win = np.full((1, 6, 6), 100, dtype=np.uint8)
    shift = np.array([[[1.0, 0.0, 4.0], [0.0, 1.0, 0.0]]])
    out = resample_windows(win, shift, 4, "linear", 9)
    assert (out[0, :, :2] == 100).all()
    assert (out[0, :, 3] == 9).all()
    with pytest.raises(ValueError, match="cubic"):
        resample_windows(win, shift, 4, "cubic", 0)


# ------------------------------------------------------------------------------- torch
@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def devices(torch):
    found = ["cpu"]
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def test_torch_gather_matches_numpy_exactly(torch):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        frames = rng.integers(0, 256, (3, 64, 72), dtype=np.uint8)
        xy = random_centres(rng, 40, (64, 72), margin=20)
        index = rng.integers(0, 3, 40)
        for pad in ("constant", "replicate"):
            ex = CropExtractor(9, pad=pad, fill=3)
            want = ex.extract(frames, xy, frame_index=index)
            for device in devices(torch):
                got = ex.extract(torch.from_numpy(frames).to(device), xy, frame_index=index)
                host = got.to_host()
                np.testing.assert_array_equal(host.crops, want.crops)
                np.testing.assert_array_equal(host.boxes, want.boxes)
                np.testing.assert_array_equal(host.valid, want.valid)
                np.testing.assert_array_equal(host.origins, want.origins)


def test_torch_accepts_the_decoder_layouts_and_tensor_centres(torch):
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (2, 50, 60), dtype=np.uint8)
    xy = np.array([[[20.0, 20.0], [30.0, 25.0]], [[10.0, 12.0], [40.0, 30.0]]], dtype=np.float32)
    ex = CropExtractor(6)
    want = ex.extract(frames, xy)
    for tensor in (torch.from_numpy(frames), torch.from_numpy(frames)[:, None]):
        got = ex.extract(tensor, torch.from_numpy(xy)).to_host()
        np.testing.assert_array_equal(got.crops, want.crops)
    single = ex.extract(torch.from_numpy(frames[0]), xy[0]).to_host()
    np.testing.assert_array_equal(single.crops, want.crops[:2])
    assert len(ex.extract(torch.from_numpy(frames[0]), xy[0])) == 2


def test_torch_input_validation_and_host_copy(torch):
    ex = CropExtractor(4)
    frame = torch.zeros((20, 20), dtype=torch.uint8)
    with pytest.raises(ValueError, match="uint8"):
        ex.extract(frame.float(), [[5.0, 5.0]])
    with pytest.raises(ValueError, match="must be"):
        ex.extract(torch.zeros((1, 2, 20, 20), dtype=torch.uint8), [[5.0, 5.0]])
    batch = ex.extract(frame, [[10.0, 10.0]])
    out = np.zeros((1, 8, 8), dtype=np.uint8)
    assert batch.to_host(out).crops is out
    with pytest.raises(ValueError, match="out must be"):
        batch.to_host(np.zeros((1, 8, 7), dtype=np.uint8))
    asynchronous = batch.to_host(non_blocking=True)
    assert asynchronous.crops.shape == (1, 8, 8)


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_torch_affine_matches_numpy_within_a_grey_level(torch, interpolation):
    rng = np.random.default_rng(2)
    frames = np.stack([smooth_frame(rng), smooth_frame(rng)])
    xy = rng.uniform(15, 110, size=(10, 2))
    angles = rng.uniform(-math.pi, math.pi, 10)
    index = rng.integers(0, 2, 10)
    for scale in (1.0, 2.0, 0.8):
        ex = CropExtractor(
            9, scale=scale, subpixel=True, interpolation=interpolation, backend="numpy", fill=4
        )
        want = ex.extract(frames, xy, angles=angles, frame_index=index)
        for device in devices(torch):
            got = ex.extract(
                torch.from_numpy(frames).to(device), xy, angles=angles, frame_index=index
            ).to_host()
            diff = np.abs(got.crops.astype(int) - want.crops.astype(int))
            if interpolation == "linear":
                assert diff.max() <= 1
            else:  # ties of the half-pixel lattice round differently
                assert diff.mean() < 8
            np.testing.assert_allclose(got.transforms, want.transforms)


def test_torch_affine_identity_and_cubic_and_edges(torch):
    rng = np.random.default_rng(3)
    frame = smooth_frame(rng, (60, 64))
    xy = np.array([[30.0, 30.0], [2.0, 3.0], [62.0, 58.0], [np.nan, np.nan]])
    ex = CropExtractor(7, fill=6)
    want = ex.extract(frame, xy)
    got = ex.extract(torch.from_numpy(frame), xy, angles=np.zeros(4)).to_host()
    np.testing.assert_array_equal(got.crops, want.crops)
    cubic = CropExtractor(7, scale=1.5, interpolation="cubic", subpixel=True)
    ref = cubic.extract(frame, xy[:1] + 0.3)
    dev = cubic.extract(torch.from_numpy(frame), xy[:1] + 0.3).to_host()
    assert np.abs(ref.crops.astype(int) - dev.crops.astype(int)).max() <= 6


def test_torch_lut_and_stretch_match_numpy(torch):
    rng = np.random.default_rng(4)
    frame = random_frame(rng, (80, 80))
    xy = rng.uniform(10, 70, size=(6, 2))
    lut = (np.arange(256) // 2).astype(np.uint8)
    ex = CropExtractor(8, lut=lut, stretch=(2.0, 98.0))
    want = ex.extract(frame, xy)
    for device in devices(torch):
        got = ex.extract(torch.from_numpy(frame).to(device), xy).to_host()
        assert np.abs(got.crops.astype(int) - want.crops.astype(int)).max() <= 1
    assert ex.extract(torch.from_numpy(frame), np.empty((0, 2))).crops.shape == (0, 16, 16)


def test_torch_stretch_chunks_large_batches(torch, monkeypatch):
    rng = np.random.default_rng(5)
    frame = random_frame(rng, (80, 80))
    xy = rng.uniform(10, 70, size=(6, 2))
    ex = CropExtractor(8, stretch=(2.0, 98.0))
    want = ex.extract(frame, xy)
    crops_torch = pytest.importorskip("naps.crops_torch")
    monkeypatch.setattr(crops_torch, "_QUANTILE_LIMIT", 300)  # two crops per quantile call
    got = ex.extract(torch.from_numpy(frame), xy).to_host()
    assert np.abs(got.crops.astype(int) - want.crops.astype(int)).max() <= 1


def test_histogram_percentiles_equal_numpy_percentile():
    rng = np.random.default_rng(21)
    crops = rng.integers(0, 256, (7, 13, 11), dtype=np.uint8)
    crops[3] = 90  # a flat crop
    for low, high in ((0.0, 100.0), (1.0, 99.0), (12.5, 87.3)):
        flat = crops.reshape(7, -1).astype(np.float64)
        lo_ref, hi_ref = np.percentile(flat, (low, high), axis=1)
        gain = 255.0 / np.maximum(hi_ref - lo_ref, 1.0)
        want = np.clip(np.floor((flat - lo_ref[:, None]) * gain[:, None] + 0.5), 0, 255)
        got = stretch_crops(crops, (low, high)).reshape(7, -1)
        assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


# ------------------------------------------------------------------------------ mosaic
def test_mosaic_layout_and_attribution():
    rng = np.random.default_rng(22)
    frame = random_frame(rng, (60, 60))
    batch = CropExtractor(5).extract(frame, rng.uniform(10, 50, size=(7, 2)))
    mosaic = batch.to_mosaic(columns=3, gap=4, value=9)
    assert mosaic.canvas.shape == (3 * 14, 3 * 14)
    assert mosaic.pitch == 14
    for i in range(7):
        x, y = mosaic.origins[i]
        np.testing.assert_array_equal(mosaic.canvas[y : y + 10, x : x + 10], batch.crops[i])
        assert (mosaic.canvas[y : y + 10, x + 10 : x + 14] == 9).all()  # the gap lane
    assert (mosaic.canvas[28:38, 14:24] == 9).all()  # unused tiles
    inside = [(mosaic.origins[i] + 4.5).tolist() for i in range(7)]
    assert mosaic.tile_of(inside).tolist() == list(range(7))
    gap_unused_outside = [
        [11.0, 4.0],
        [18.0, 32.0],
        [-1.0, 3.0],
        [200.0, 3.0],
        [3.0, 500.0],
        [43.0, 3.0],
    ]
    assert mosaic.tile_of(gap_unused_outside).tolist() == [-1] * 6
    np.testing.assert_allclose(
        mosaic.tile_points(4, [[mosaic.origins[4, 0] + 2, mosaic.origins[4, 1] + 3]]), [[2, 3]]
    )


def test_mosaic_default_columns_and_validation():
    frame = random_frame(np.random.default_rng(23), (60, 60))
    batch = CropExtractor(5).extract(frame, np.full((16, 2), 30.0))
    mosaic = batch.to_mosaic()
    assert mosaic.columns == 4
    assert mosaic.canvas.shape == (4 * 18, 4 * 18)
    empty = CropExtractor(5).extract(frame, np.empty((0, 2))).to_mosaic()
    assert empty.canvas.shape == (18, 18)
    assert empty.tile_of([[1.0, 1.0]]).tolist() == [-1]
    with pytest.raises(ValueError, match="columns"):
        batch.to_mosaic(columns=0)
    with pytest.raises(ValueError, match="columns"):
        batch.to_mosaic(gap=-1)
