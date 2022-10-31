from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import delta
from delta import data

# fmt: off
MASK_1D = np.array(
    [
        [
            0, 1,
            0, 0, 0, 1, 1, 0, 1, 1,
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        ]
    ]
)
DISTS_1D = np.array(
    [
        [
            6, 0,
            4, 4, 4, 0, 0, 2, 0, 0,
            5, 6, 6, 6, 4, 0, 2, 0, 3, 3, 0, 0, 0, 3, 3, 0, 2, 0, 0, 0, 2, 0, 0, 0,
        ]
    ]
)
# fmt: on


def test_seg_weights_simple():
    classweights = (0.3, 0.7)
    w0 = 12.0
    sigma = 2.0
    weights = delta.data.seg_weights(
        MASK_1D, classweights=classweights, w0=w0, sigma=sigma
    )
    target = w0 * np.exp(-(DISTS_1D**2) / (2 * sigma**2))
    target[DISTS_1D == 0] = classweights[1]
    target[DISTS_1D != 0] += classweights[0]
    npt.assert_allclose(weights, target)


def test_seg_weights_complex():
    path = Path(__file__).parent / "data/images"
    mask = delta.utils.read_image(path / "complex_mask.png")
    target = delta.utils.read_image(path / "complex_weights.png")
    weights = delta.data.seg_weights(mask)
    norm_weights = weights / weights.max()
    int_weights = delta.utils.to_integer_values(norm_weights, np.uint8)
    npt.assert_allclose(int_weights / 255, target)


def test_seg_weights_2D_complex():
    path = Path(__file__).parent / "data/images"
    mask = delta.utils.read_image(path / "complex_mask.png")
    target = delta.utils.read_image(path / "complex_weights_2D.png")
    weights = delta.data.seg_weights_2D(mask, classweights=(0.3, 0.7))
    norm_weights = weights / weights.max()
    int_weights = delta.utils.to_integer_values(norm_weights, np.uint8)
    npt.assert_allclose(int_weights / 255, target)


def test_postprocess_binarize():
    logits = np.array([[-3, -1, 1.5], [3, 0.5, -0.1]])
    mask = data.postprocess(logits, crop=True)
    npt.assert_array_equal(mask, np.array([[0, 0, 1], [1, 1, 0]]))


def test_postprocess_minsize():
    preds = np.array(
        [
            [1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ],
        dtype=np.float32
    )
    mask = data.postprocess(preds, crop=True, min_size=4)
    minsize = np.array(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ]
    )
    npt.assert_array_equal(mask, minsize)


def test_postprocess_opening():
    preds = np.array(
        [
            [1, 0, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
        ],
        dtype=np.float32
    )
    mask = data.postprocess(preds, crop=False, min_size=None, square_size=3)
    minsize = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
        ]
    )
    npt.assert_array_equal(mask, minsize)


@pytest.mark.parametrize(
    "shiftX, shiftY",
     [
         (0.6, 1.6),
         (0.6, 2.0),
         (0.6, 2.4),
         (1.0, 1.6),
         (1.0, 2.0),
         (1.0, 2.4),
         (1.4, 1.6),
         (1.4, 2.0),
         (1.4, 2.4),
     ])
def test_zoomshift_shift_order0(shiftX, shiftY):
    img = np.array(
        [
            [1, 0, 1, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0],
        ],
        dtype=np.uint8
    )
    actual = data.zoomshift(img, zoomlevel=1.0, shiftX=shiftX/img.shape[0], shiftY=shiftY/img.shape[1], order=0)
    # 1 up, 2 left
    truth = np.array(
        [
            [1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ],
    )
    npt.assert_array_equal(actual, truth)


def test_zoomshift_zoom_order0():
    img = np.array(
        [
            [1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.uint8
    )
    actual = data.zoomshift(img, zoomlevel=2.0, shiftX=0, shiftY=0, order=0)
    truth = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    npt.assert_array_equal(actual, truth)


def test_zoomshift_zoomshift_order0():
    img = np.array(
        [
            [1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.uint8
    )
    actual = data.zoomshift(img, zoomlevel=2.0, shiftX=1.0/img.shape[0], shiftY=2.0/img.shape[1], order=0)
    truth = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1],
        ]
    )
    npt.assert_array_equal(actual, truth)
