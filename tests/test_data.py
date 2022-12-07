from pathlib import Path

import numpy as np

import delta

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
    weights = delta.data.seg_weights(MASK_1D, classweights=classweights, w0=w0, sigma=sigma)
    target = w0 * np.exp(-DISTS_1D**2 / (2 * sigma**2))
    target[DISTS_1D == 0] = classweights[1]
    target[DISTS_1D != 0] += classweights[0]
    assert np.allclose(weights, target)


def test_seg_weights_complex():
    path = Path(__file__).parent / "data/images"
    mask = delta.utils.read_image(path / "complex_mask.png")
    target = delta.utils.read_image(path / "complex_weights.png")
    weights = delta.data.seg_weights(mask)
    norm_weights = weights / weights.max()
    int_weights = delta.utils.to_integer_values(norm_weights, np.uint8)
    assert np.allclose(int_weights / 255, target)


def test_seg_weights_2D_complex():
    path = Path(__file__).parent / "data/images"
    mask = delta.utils.read_image(path / "complex_mask.png")
    target = delta.utils.read_image(path / "complex_weights_2D.png")
    weights = delta.data.seg_weights_2D(mask, classweights=(0.3, 0.7))
    norm_weights = weights / weights.max()
    int_weights = delta.utils.to_integer_values(norm_weights, np.uint8)
    assert np.allclose(int_weights / 255, target)
