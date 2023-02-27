#!/usr/bin/env python3
"""
Created on Tue Oct 19 15:01:05 2021.

@author: ooconnor
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import numpy.random as npr
import pytest
from scipy.ndimage import binary_fill_holes
from scipy.special import logit

import delta
from delta import pipeline, utils
from delta.config import DEFAULT_CONFIG_2D
from delta.lineage import CellFeatures


@pytest.mark.parametrize("xtl", [-1, 0])
@pytest.mark.parametrize("xbr", [20, 21])
@pytest.mark.parametrize("ytl", [-1, 0])
@pytest.mark.parametrize("ybr", [10, 11])
def test_croppingbox_full_img_crop_patch(xtl, xbr, ytl, ybr):
    rng = npr.default_rng(0)
    img = rng.random((10, 20))
    box = utils.CroppingBox(xtl=xtl, xbr=xbr, ytl=ytl, ybr=ybr)
    patch = box.crop(img)
    np.testing.assert_array_equal(patch.shape, (box.ybr - box.ytl, box.xbr - box.xtl))
    patched_img = box.patch(np.zeros_like(img), patch)
    np.testing.assert_array_equal(patched_img, img)


@pytest.mark.parametrize(
    "xtl, xbr, ytl, ybr",
    [
        # inside
        (2, 16, 3, 7),
        # corners
        (-5, 4, -10, 2),
        (5, 24, -10, 2),
        (-5, 4, 6, 22),
        (5, 24, 6, 22),
        # edges
        (-5, 24, -10, 2),
        (-5, 24, 6, 22),
        (-5, 4, -10, 22),
        (5, 24, -10, 22),
    ],
)
def test_croppingbox_partial_crop_patch(xtl, xbr, ytl, ybr):
    rng = npr.default_rng(0)
    img = rng.random((10, 20))
    box = utils.CroppingBox(xtl=xtl, xbr=xbr, ytl=ytl, ybr=ybr)
    patch = box.crop(img)
    np.testing.assert_array_equal(patch.shape, (box.ybr - box.ytl, box.xbr - box.xtl))
    patched_img = box.patch(np.array(img), patch)
    np.testing.assert_array_equal(patched_img, img)


# %% Test growth rate


def test_growthrate_onecell():
    # Cell 0 is on frames 0, 1, 2
    pole = np.zeros(2, dtype=np.int16)
    feats = CellFeatures(
        old_pole=pole,
        new_pole=pole,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    lineage = utils.Lineage()
    lineage.update(cell=None, frame=0, attrib=[0], features=[feats])
    lineage.update(cell=0, frame=1, attrib=[0], features=[feats])
    lineage.update(cell=0, frame=2, attrib=[0], features=[feats])
    lineage.setvalue(0, frame=0, feature="length", value=1.0)
    lineage.setvalue(0, frame=1, feature="length", value=1.1)
    lineage.setvalue(0, frame=2, feature="length", value=1.2)
    assert lineage.growthrate(0, frame=0, feature="length") == np.log(1.1 / 1.0)
    assert lineage.growthrate(0, frame=1, feature="length") == np.log(1.2 / 1.0) / 2.0
    assert lineage.growthrate(0, frame=2, feature="length") == np.log(1.2 / 1.1)


def test_growthrate_division():
    # Cell 0 is on frames 0, 1, 2
    # Cell 1, born from cell 0 on frame 1, is on frames 1 and 2
    pole = np.zeros(2, dtype=np.int16)
    lineage = utils.Lineage()
    feats = CellFeatures(
        old_pole=pole,
        new_pole=pole,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    lineage.update(cell=None, frame=0, attrib=[0], features=[feats])
    lineage.update(cell=0, frame=1, attrib=[1, 0], features=[feats, feats])
    lineage.update(cell=0, frame=2, attrib=[0], features=[feats])
    lineage.update(cell=1, frame=2, attrib=[1], features=[feats])
    lineage.setvalue(0, frame=0, feature="length", value=2.0)
    lineage.setvalue(0, frame=1, feature="length", value=1.2)
    lineage.setvalue(0, frame=2, feature="length", value=1.3)
    lineage.setvalue(1, frame=1, feature="length", value=1.0)
    lineage.setvalue(1, frame=2, feature="length", value=1.1)
    assert lineage.growthrate(0, frame=0, feature="length") == np.log((1.2 + 1.0) / 2.0)
    assert (
        lineage.growthrate(0, frame=1, feature="length")
        == np.log(1.3 / (2.0 * 1.2 / (1.2 + 1.0))) / 2.0
    )
    assert lineage.growthrate(0, frame=2, feature="length") == np.log(1.3 / 1.2)
    assert (
        lineage.growthrate(1, frame=1, feature="length")
        == np.log(1.1 / (2.0 * 1.0 / (1.2 + 1.0))) / 2.0
    )
    assert lineage.growthrate(1, frame=2, feature="length") == np.log(1.1 / 1.0)


def test_growthrate_division_last():
    # Cell 0 is on frames 0 and 1
    # Cell 1, born from cell 0 on frame 1, is on frame 1
    pole = np.zeros(2, dtype=np.int16)
    feats = CellFeatures(
        old_pole=pole,
        new_pole=pole,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    lineage = utils.Lineage()
    lineage.update(cell=None, frame=0, attrib=[0], features=[feats])
    lineage.update(cell=0, frame=1, attrib=[1, 0], features=[feats, feats])
    lineage.setvalue(0, frame=0, feature="length", value=2.0)
    lineage.setvalue(0, frame=1, feature="length", value=1.2)
    lineage.setvalue(1, frame=1, feature="length", value=1.0)
    assert lineage.growthrate(0, frame=0, feature="length") == np.log((1.2 + 1.0) / 2.0)
    assert lineage.growthrate(0, frame=1, feature="length") == np.log(
        1.2 / (2.0 * 1.2 / (1.2 + 1.0))
    )
    assert lineage.growthrate(1, frame=1, feature="length") == np.log(
        1.0 / (2.0 * 1.0 / (1.2 + 1.0))
    )


def test_growthrate_double_division():
    # Cell 0 is on frames 0, 1, 2
    # Cell 1, born from cell 0 on frame 1, is on frames 1 and 2
    # Cell 2, born from cell 1 on frame 2, is on frame 2
    # Cell 3, born from cell 0 on frame 2, is on frame 2
    pole = np.zeros(2, dtype=np.uint16)
    feats = CellFeatures(
        old_pole=pole,
        new_pole=pole,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    lineage = utils.Lineage()
    lineage.update(cell=None, frame=0, attrib=[0], features=[feats])
    lineage.update(cell=0, frame=1, attrib=[1, 0], features=[feats, feats])
    lineage.update(cell=1, frame=2, attrib=[1, 0], features=[feats, feats])
    lineage.update(cell=0, frame=2, attrib=[2, 3], features=[feats, feats])
    lineage.setvalue(0, frame=0, feature="length", value=2.0)
    lineage.setvalue(0, frame=1, feature="length", value=1.2)
    lineage.setvalue(0, frame=2, feature="length", value=0.8)
    lineage.setvalue(1, frame=1, feature="length", value=1.0)
    lineage.setvalue(1, frame=2, feature="length", value=0.7)
    lineage.setvalue(2, frame=2, feature="length", value=0.4)
    lineage.setvalue(3, frame=2, feature="length", value=0.5)
    assert lineage.growthrate(0, frame=0, feature="length") == np.log((1.2 + 1.0) / 2.0)
    assert (
        lineage.growthrate(0, frame=1, feature="length")
        == np.log((0.8 + 0.5) / (2.0 * 1.2 / (1.2 + 1.0))) / 2.0
    )
    assert lineage.growthrate(0, frame=2, feature="length") == np.log(
        0.8 / (1.2 * 0.8 / (0.8 + 0.5))
    )
    assert (
        lineage.growthrate(1, frame=1, feature="length")
        == np.log((0.7 + 0.4) / (2.0 * 1.0 / (1.2 + 1.0))) / 2.0
    )
    assert lineage.growthrate(1, frame=2, feature="length") == np.log(
        0.7 / (1.0 * 0.7 / (0.7 + 0.4))
    )
    assert lineage.growthrate(2, frame=2, feature="length") == np.log(
        0.4 / (1.0 * 0.4 / (0.7 + 0.4))
    )
    assert lineage.growthrate(3, frame=2, feature="length") == np.log(
        0.5 / (1.2 * 0.5 / (0.8 + 0.5))
    )


# %% Test deskew


@pytest.mark.parametrize(
    "filename, angle0", [("angle_+0.53.tif", 0.53), ("angle_-0.32.tif", -0.32)]
)
def test_deskew(filename, angle0):
    file = Path(__file__).parent / "data/images" / filename
    image0 = utils.read_image(file)
    image = utils.imrotate(image0, -angle0)
    assert np.abs(utils.deskew(image)) <= 0.25
    for angle in np.linspace(-3, 3, 30):
        image = utils.imrotate(image0, angle - angle0)
        assert np.abs(utils.deskew(image) + angle) <= 0.25


# %% Eval create_windows


@pytest.mark.parametrize(
    "image_size, nb_tiles",
    [
        ((512, 512), 1),
        ((513, 512), 2),
        ((1000, 512), 2),
        ((1001, 512), 3),
        ((2048, 2048), 25),
        ((200, 200), 1),
        ((600, 200), 2),
        ((696, 520), 4),
    ],
)
def test_create_windows_number(image_size, nb_tiles):
    target_size = (512, 512)
    img = np.zeros(image_size)
    windows, _, _ = utils.create_windows(img, target_size=target_size, min_overlap=24)
    assert windows.shape == (nb_tiles, *target_size)
    windows, _, _ = utils.create_windows(img.T, target_size=target_size, min_overlap=24)
    assert windows.shape == (nb_tiles, *target_size)


# fmt: off
@pytest.mark.parametrize(
    "image_size, target_size, min_overlap, nb_tiles",
    [
        (10, 1, 0, 10),

        (10, 2, 0, 5), (10, 2, 1, 9),

        (10, 3, 0, 4), (10, 3, 1, 5), (10, 3, 2, 8),

        (10, 4, 0, 3), (10, 4, 1, 3), (10, 4, 2, 4), (10, 4, 3, 7),

        (10, 5, 0, 2), (10, 5, 1, 3), (10, 5, 2, 3), (10, 5, 3, 4),
        (10, 5, 4, 6),

        (10, 6, 0, 2), (10, 6, 1, 2), (10, 6, 2, 2), (10, 6, 3, 3),
        (10, 6, 4, 3), (10, 6, 5, 5),

        (10, 7, 0, 2), (10, 7, 1, 2), (10, 7, 2, 2), (10, 7, 3, 2),
        (10, 7, 4, 2), (10, 7, 5, 3), (10, 7, 6, 4),

        (10, 8, 0, 2), (10, 8, 1, 2), (10, 8, 2, 2), (10, 8, 3, 2),
        (10, 8, 4, 2), (10, 8, 5, 2), (10, 8, 6, 2), (10, 8, 7, 3),

        (10, 9, 0, 2), (10, 9, 1, 2), (10, 9, 2, 2), (10, 9, 3, 2),
        (10, 9, 4, 2), (10, 9, 5, 2), (10, 9, 6, 2), (10, 9, 7, 2),
        (10, 9, 8, 2),

        (10, 10, 0, 1), (10, 10, 1, 1), (10, 10, 2, 1), (10, 10, 3, 1),
        (10, 10, 4, 1), (10, 10, 5, 1), (10, 10, 6, 1), (10, 10, 7, 1),
        (10, 10, 8, 1), (10, 10, 9, 1),
    ]
)
# fmt: on
def test_create_windows_number_1d(image_size, target_size, min_overlap, nb_tiles):
    other_dim = 20
    image = np.zeros((other_dim, image_size))
    windows, loc_y, loc_x = utils.create_windows(
        image, target_size=(other_dim, target_size), min_overlap=min_overlap
    )
    assert len(windows) == nb_tiles
    assert len(loc_x) == nb_tiles
    np.testing.assert_array_equal(loc_y, [(0, other_dim)])
    windows, loc_y, loc_x = utils.create_windows(
        image.T, target_size=(target_size, other_dim), min_overlap=min_overlap
    )
    assert len(windows) == nb_tiles
    assert len(loc_y) == nb_tiles
    np.testing.assert_array_equal(loc_x, [(0, other_dim)])


def test_create_windows():
    image = np.arange(90).reshape((9, 10))
    windows, loc_y, loc_x = utils.create_windows(
        image, target_size=(5, 3), min_overlap=1
    )
    pattern = np.arange(3)[np.newaxis, :] + 10 * np.arange(5)[:, np.newaxis]
    truth = (
        np.array([0, 1, 3, 5, 7, 40, 41, 43, 45, 47])[:, np.newaxis, np.newaxis]
        + pattern[np.newaxis, :, :]
    )
    np.testing.assert_array_equal(windows, truth)
    np.testing.assert_array_equal(loc_y, [(i, i + 5) for i in [0, 4]])
    np.testing.assert_array_equal(loc_x, [(i, i + 3) for i in [0, 1, 3, 5, 7]])


def test_create_windows_pad():
    image = np.arange(20).reshape((2, 10))
    windows, loc_y, loc_x = utils.create_windows(
        image, target_size=(6, 10), min_overlap=1
    )
    truth = np.array(
        [[0] * 10, [0] * 10, image[0, :], image[1, :], [0] * 10, [0] * 10],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(windows, [truth])
    np.testing.assert_array_equal(loc_y, [(-2, 4)])
    np.testing.assert_array_equal(loc_x, [(0, 10)])


@pytest.mark.parametrize(
    "loc_y, loc_x, expected",
    [
        (
            [(0, 6), (4, 10), (8, 14)],
            [(0, 6), (2, 8)],
            [
                *[[1] * 4 + [2] * 4] * 5,
                *[[3] * 4 + [4] * 4] * 4,
                *[[5] * 4 + [6] * 4] * 5,
            ],
        ),
    ],
)
def test_stitch_pic(expected, loc_y, loc_x):
    windows = np.zeros((len(loc_y) * len(loc_x), 6, 6), dtype=np.uint8)
    for i in range(6):
        windows[i, :, :] = i + 1
    img = utils.stitch_pic(windows, loc_y, loc_x)
    np.testing.assert_array_equal(img, expected)


@pytest.mark.parametrize(
    "image_shape",
    [
        (512, 512),
        (513, 512),
        (1000, 512),
        (1001, 512),
        (2048, 2048),
        (200, 200),
        (600, 200),
        (696, 520),
    ],
)
def test_create_windows_stitch_pic(image_shape):
    rng = npr.default_rng(0)
    image = rng.random(image_shape)
    windows, loc_y, loc_x = utils.create_windows(
        image, target_size=(512, 512), min_overlap=24
    )
    stitched_image = utils.stitch_pic(windows, loc_y, loc_x)
    # Currently need this cropping when the image is smaller than the target size
    stitched_image = stitched_image[0 : image_shape[0], 0 : image_shape[1]]
    np.testing.assert_array_equal(stitched_image, image)


@pytest.mark.parametrize(
    "solution",
    [
        np.array(
            [
                [1, 1, 1, 0, 2, 2],
                [1, 1, 1, 0, 0, 2],
                [0, 0, 0, 0, 2, 2],
                [3, 3, 0, 0, 0, 0],
                [3, 3, 0, 5, 5, 0],
                [0, 0, 0, 0, 5, 0],
                [0, 4, 4, 0, 5, 5],
                [0, 0, 0, 0, 0, 5],
            ],
            dtype=np.uint16,
        ),
        np.array(
            [
                [2, 2, 0, 0, 0],
                [2, 2, 0, 1, 1],
                [2, 2, 0, 0, 0],
                [0, 2, 0, 0, 3],
                [0, 0, 0, 0, 3],
                [4, 4, 4, 0, 3],
                [0, 0, 4, 0, 0],
                [6, 0, 0, 0, 5],
                [6, 6, 0, 0, 0],
            ],
            dtype=np.uint16,
        ),
    ],
)
def test_label_seg(solution):
    mask = (solution > 0).astype(np.uint8)
    np.testing.assert_array_equal(utils.label_seg(mask), solution)
    labels = [int(x) for x in npr.default_rng(seed=1).choice(1000, solution.max())]
    for oldlabel, newlabel in enumerate(labels):
        solution[solution == oldlabel + 1] = newlabel
    assert np.array_equal(utils.label_seg(mask, cellnumbers=labels), solution)


def test_cell_perimeter():
    mask = np.array(
        [[0, 1, 1], [1, 1, 0], [0, 1, 0]],
        dtype=np.uint8,
    )
    perimeter = 2 + 3 * np.sqrt(2)
    contours = utils.find_contours(mask)
    np.testing.assert_allclose(utils.cell_perimeter(contours[0]), perimeter)


# %% Eval find_contours


def rand_mask(image_size, max_dots=100, max_radius=30, seed=None):
    """Generate a random binary mask with random size dots."""
    rng = npr.default_rng(seed)
    mask = np.zeros(image_size, dtype=np.uint8)
    num_dots = rng.integers(max_dots, endpoint=True)

    for dot in range(num_dots):
        cv2.circle(
            mask,
            center=(rng.integers(image_size[0]), rng.integers(image_size[0])),
            radius=rng.integers(1, max_radius),
            color=1,
            thickness=-1,
        )

    # Holes cause issues here
    mask = binary_fill_holes(mask).astype(np.uint8)

    return mask


def evaluate_contours(mask, contours, image_size):
    """Reconstruct mask from contours and make it is the same as the original."""
    new_mask = np.zeros(image_size, dtype=np.uint8)
    new_mask = cv2.drawContours(new_mask, contours, -1, 1, thickness=-1)

    return not (np.bitwise_xor(mask, new_mask).any())


@pytest.mark.parametrize("seed", range(10))  # Run it 10 times
@pytest.mark.parametrize(
    "image_size",
    [
        (512, 512),
        (2048, 2048),
        (200, 200),
        (200, 600),
        (600, 200),
        (100, 700),
        (696, 520),
        (256, 32),
    ],
)
def test_find_contours(image_size, seed):
    mask = rand_mask(image_size, seed=seed)
    contours = utils.find_contours(mask)
    assert evaluate_contours(mask, contours, image_size)


def test_skeleton_poles():
    skel = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    mask_poles = utils.skeleton_poles(skel)
    solution = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask_poles, solution)


def test_skeleton_poles_singular():
    skel = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    mask_poles = utils.skeleton_poles(skel)
    solution = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask_poles, solution)


def test_extract_poles():
    labels = np.array(
        [
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 4],
            [3, 3, 3, 0, 4, 4],
        ]
    )
    skel = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    mask_poles = utils.skeleton_poles(skel)
    poles = utils.extract_poles(mask_poles, labels)
    solution = {
        0: [np.array([0, 1]), np.array([2, 1])],
        1: [np.array([0, 3]), np.array([1, 5])],
        2: [np.array([4, 0]), np.array([4, 2])],
        3: [np.array([3, 4]), np.array([4, 5])],
    }
    assert poles.keys() == solution.keys()
    for k in solution.keys():
        assert np.array_equal(poles[k], solution[k])


def test_extract_poles_not2():
    labels = np.array(
        [
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 3, 3],
            [1, 1, 1, 0, 3, 3],
        ]
    )
    skel = np.array(
        [
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    mask_poles = utils.skeleton_poles(skel)
    poles = utils.extract_poles(mask_poles, labels)
    solution = {
        0: [np.array([0, 1]), np.array([2, 0]), np.array([4, 2])],
        1: [np.array([0, 4]), np.array([1, 5])],
        2: [np.array([3, 4])],
    }
    assert poles.keys() == solution.keys()
    for k in solution.keys():
        assert np.array_equal(poles[k], solution[k])


def test_extract_poles_missing():
    labels = np.array(
        [
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 5],
            [3, 3, 3, 0, 5, 5],
        ]
    )
    skel = np.array(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    mask_poles = utils.skeleton_poles(skel)
    poles = utils.extract_poles(mask_poles, labels)
    solution = {
        0: [np.array([0, 1]), np.array([2, 1])],
        1: [np.array([0, 3]), np.array([1, 5])],
        2: [np.array([4, 0]), np.array([4, 2])],
        4: [np.array([3, 4]), np.array([4, 5])],
    }
    assert poles.keys() == solution.keys()
    for k in solution.keys():
        assert np.array_equal(poles[k], solution[k])


def test_extract_poles_loop():
    labels = np.array(
        [
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 4, 4],
            [3, 3, 3, 0, 4, 4],
        ]
    )
    skel = np.array(
        [
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
        ],
        dtype=np.bool_,
    )
    mask_poles = utils.skeleton_poles(skel)
    poles = utils.extract_poles(mask_poles, labels)
    solution = {
        1: [np.array([0, 4]), np.array([1, 5])],
        2: [np.array([4, 0]), np.array([4, 2])],
        3: [np.array([3, 4]), np.array([4, 5])],
    }
    assert poles.keys() == solution.keys()
    for k in solution.keys():
        assert np.array_equal(poles[k], solution[k])


def test_getpoles():
    labels = np.array(
        [
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 3, 3],
            [1, 1, 1, 0, 3, 3],
            [0, 0, 0, 0, 3, 3],
            [0, 4, 4, 4, 0, 3],
            [0, 4, 0, 4, 0, 0],
            [0, 4, 4, 4, 0, 0],
        ]
    )
    poles = utils.getpoles(labels > 0, labels)
    solution = {
        0: [np.array([0, 0]), np.array([4, 2])],
        1: [np.array([0, 4]), np.array([1, 5])],
        2: [np.array([3, 4]), np.array([6, 5])],
        3: [np.array([6, 1]), np.array([8, 3])],
    }
    assert poles.keys() == solution.keys()
    for k in solution.keys():
        assert np.array_equal(poles[k], solution[k])


@pytest.mark.parametrize("seed", range(1000))  # Run it 1000 times
def test_track_poles(seed):
    rng = npr.default_rng(seed)
    # Random previous old pole:
    prev_old = rng.integers(low=0, high=2**15, size=2, dtype=np.int16)
    # Random previous new pole is some small random distance away:
    while True:
        increment = rng.normal(loc=0, scale=20, size=2).astype(np.int16)
        dist = np.linalg.norm(increment)
        if dist < 10:  # At least 10 pixels between poles
            continue
        prev_new = prev_old + increment
        if np.all(prev_new >= 0):
            break

    while True:
        # Simulate whole cell shift:
        cell_shift = rng.gamma(shape=2, scale=2, size=2)

        # Simulate small pole shifts:
        increment = rng.uniform(low=-3, high=3, size=2)
        new_new = prev_new + cell_shift + increment

        # Simulate small pole shifts:
        increment = rng.uniform(low=-3, high=3, size=2)
        new_old = prev_old + cell_shift + increment

        new_new = new_new.astype(np.int16)
        new_old = new_old.astype(np.int16)
        if np.all(new_new >= 0) and np.all(new_old >= 0):
            break

    features = CellFeatures(
        old_pole=new_old,
        new_pole=new_new,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    if rng.random() < 0.5:
        features.swap_poles()

    feats = utils.track_poles(features, prev_old, prev_new)

    assert np.all(feats.old_pole == new_old)
    assert np.all(feats.new_pole == new_new)


@pytest.mark.parametrize("seed", range(1000))  # Run it 1000 times
def test_division_poles(seed):
    rng = npr.default_rng(seed)
    # Random previous old pole:
    prev_old = rng.integers(low=0, high=2**15, size=2, dtype=np.int16)
    # Random previous new pole is some small random distance away:
    while True:
        increment = rng.normal(loc=0, scale=20, size=2).astype(np.int16)
        dist = np.linalg.norm(increment)
        if dist < 20:  # At least 20 pixels between poles
            continue
        prev_new = prev_old + increment
        if np.all(prev_new >= 0):
            break

    # Generate shift poles to create "current frame"
    while True:
        # Simulate whole cell shift:
        cell_shift = rng.gamma(shape=2, scale=2, size=2)

        # Simulate small pole shifts:
        increment = rng.uniform(low=-3, high=3, size=2)
        daughter_old = prev_new + cell_shift + increment

        # Simulate small pole shifts:
        increment = rng.uniform(low=-3, high=3, size=2)
        mother_old = prev_old + cell_shift + increment

        daughter_old = daughter_old.astype(np.int16)
        mother_old = mother_old.astype(np.int16)
        if np.all(daughter_old >= 0) and np.all(mother_old >= 0):
            break

    # Generate septum and new poles:
    septum = mother_old / 2 + daughter_old / 2

    # Shift by 2 pixels towards mother to get mother's new pole
    sep2mot = 2 * (septum - mother_old) / np.linalg.norm(septum - mother_old)
    mother_new = (mother_old + sep2mot).astype(np.int16)

    # Shift by 2 pixels towards mother to get mother's new pole
    sep2dau = 2 * (septum - daughter_old) / np.linalg.norm(septum - daughter_old)
    daughter_new = (daughter_old + sep2dau).astype(np.int16)

    # Randomly assign poles to the input lists:
    mother = CellFeatures(
        old_pole=mother_old,
        new_pole=mother_new,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    if rng.random() < 0.5:
        mother.swap_poles()

    daughter = CellFeatures(
        old_pole=daughter_old,
        new_pole=daughter_new,
        length=0.0,
        width=0.0,
        area=0.0,
        perimeter=0.0,
        fluo=[],
        edges="",
    )
    if rng.random() < 0.5:
        daughter.swap_poles()

    if rng.random() > 0.5:
        first_cell_is_mother_t = True
        features1 = mother
        features2 = daughter
    else:
        first_cell_is_mother_t = False
        features2 = mother
        features1 = daughter

    # Run division poles:
    mother_out, daughter_out, first_cell_is_mother = utils.division_poles(
        features1, features2, prev_old, prev_new
    )

    # Assertions:
    assert first_cell_is_mother == first_cell_is_mother_t
    assert np.all(mother_old == mother_out.old_pole)
    assert np.all(mother_new == mother_out.new_pole)
    assert np.all(daughter_old == daughter_out.old_pole)
    assert np.all(daughter_new == daughter_out.new_pole)


# %% Make sure contour coloring and cv2 colormap work
def _base_test_color(max_dots, seed=None):
    # Get random mask and RGB frame:
    mask = rand_mask((2048, 2048), max_dots=max_dots, seed=seed)
    frame = np.repeat(mask[:, :, np.newaxis].astype(np.float64), 3, axis=2)

    # Get cells numbers and contours:
    cells, contours = utils.getcellsinframe(utils.label_seg(mask), return_contours=True)

    # Get random colors:
    colors = utils.getrandomcolors(cells, seed=seed)

    # Color cells contours:
    for c, cell in enumerate(cells):
        frame = cv2.drawContours(
            frame,
            contours,
            c,
            color=colors[cell],
            thickness=1,
        )

    return mask, frame


def _evaluate_color_nb(mask, frame):
    # Get cell contours:
    contours = utils.find_contours(mask)

    # Get colors present in frame:
    colors = np.unique(
        np.reshape(frame, (frame.shape[0] * frame.shape[1], frame.shape[2])), axis=0
    )

    # Assert that we have the right nb of colors:
    assert colors.shape[0] == len(contours) + np.unique(mask).shape[0]


@pytest.mark.parametrize("seed", range(5))  # Run it 5 times
@pytest.mark.parametrize("max_dots", [0, 1, 2, 100])
def test_color_contours(max_dots, seed):
    mask, frame = _base_test_color(0, seed=seed)
    _evaluate_color_nb(mask, frame)


# %% Test XPReader


class Test_XPReader:
    def test_2D_tif_filename_noresfolder(self):
        # Init reader
        xpreader = utils.XPReader(Path(__file__).parent / "data/movie_2D_tif")

        # Init pipeline
        pipeline.Pipeline(xpreader, DEFAULT_CONFIG_2D)

    def test_2D_tif_allargs(self):
        # Init reader
        xpreader = utils.XPReader(
            Path(__file__).parent / "data/movie_2D_tif",
            prototype="Position%02dChannel%02dFrame%06d.tif",
            fileorder="pct",
            filenamesindexing=1,
        )

        # Init pipeline:
        pipeline.Pipeline(
            xpreader,
            DEFAULT_CONFIG_2D,
            resfolder=Path(__file__).parent / "data/movie_2D_tif/results",
        )

    def test_2D_tif_xpreader_filename_noresfolder(self):
        # Init reader
        xpreader = utils.XPReader(Path(__file__).parent / "data" / "movie_2D_tif")

        # Init pipeline:
        pipeline.Pipeline(xpreader, DEFAULT_CONFIG_2D)

    def test_2D_tif_xpreader_filename_and_resfolder(self):
        # Init reader
        xpreader = utils.XPReader(Path(__file__).parent / "data" / "movie_2D_tif")

        # Init pipeline:
        pipeline.Pipeline(
            xpreader,
            DEFAULT_CONFIG_2D,
            resfolder=Path(__file__).parent / "data" / "movie_2D_tif" / "results",
        )

    def test_moma_tif_xpreader_filename_noresfolder(self):
        # Init reader
        xpreader = utils.XPReader(
            Path(__file__).parent / "data" / "movie_mothermachine_tif"
        )

        # Init pipeline:
        pipeline.Pipeline(xpreader, DEFAULT_CONFIG_2D)


# fmt: off
UNTRACKED_LABELS = np.array(
    [
        [0, 0, 1, 0],
        [0, 2, 1, 0],
        [0, 2, 1, 0],
        [0, 2, 3, 3],
    ],
    dtype=np.uint16,
)
# fmt: on

TRACKING_OUTPUTS = logit(
    np.array(
        [
            [  # Cell tracked to label 2
                [0.00, 0.06, 0.00, 0.00],
                [0.02, 0.72, 0.10, 0.01],
                [0.03, 0.68, 0.06, 0.02],
                [0.12, 0.89, 0.04, 0.00],
            ],
            [  # Cell tracked to label 1
                [0.00, 0.07, 0.43, 0.00],
                [0.00, 0.02, 0.78, 0.13],
                [0.01, 0.10, 0.82, 0.07],
                [0.00, 0.25, 0.92, 0.04],
            ],
        ]
    )
)


def test_getTrackingScores_no_old_cells():
    labels = UNTRACKED_LABELS
    outputs = np.zeros((0, labels.shape[0], labels.shape[1]), dtype=np.float32)
    scores = utils.getTrackingScores(labels, outputs)
    np.testing.assert_allclose(scores, np.zeros((0, 3), dtype=np.float32))


def test_getTrackingScores_no_new_cells():
    labels = 0 * UNTRACKED_LABELS
    outputs = TRACKING_OUTPUTS
    scores = utils.getTrackingScores(labels, outputs)
    np.testing.assert_allclose(scores, np.zeros((2, 0), dtype=np.float32))


def test_getTrackingScores_no_hits():
    labels = UNTRACKED_LABELS
    outputs = TRACKING_OUTPUTS - 100
    scores = utils.getTrackingScores(labels, outputs)
    np.testing.assert_allclose(scores, [[0, 0, 0], [0, 0, 0]])


def test_getTrackingScores():
    labels = UNTRACKED_LABELS
    outputs = TRACKING_OUTPUTS
    scores = utils.getTrackingScores(labels, outputs)
    np.testing.assert_allclose(scores, [[2 / 3, 1, 0], [1, 2 / 3, 1 / 2]])


def test_cell_fluo():
    fluo_frames = np.array(
        [
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ],
            [
                [100, 101, 102, 103, 104],
                [105, 106, 107, 108, 109],
                [110, 111, 112, 113, 114],
                [115, 116, 117, 118, 119],
            ],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [False, True, True, True, False],
            [False, False, True, False, False],
            [False, False, True, True, False],
            [False, True, True, False, False],
        ],
        dtype=bool,
    )
    box = utils.CroppingBox.full(mask)
    fluos = utils.cell_fluo(fluo_frames, mask, box)
    expected = (1 + 2 + 3 + 7 + 12 + 13 + 16 + 17) / 8
    np.testing.assert_allclose(fluos, [expected, expected + 100])
