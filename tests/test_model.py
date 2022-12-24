from pathlib import Path

import cv2
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

import delta


def nb_params_unet(filters: [int]) -> int:
    nb_params = 0
    for filters_a, filters_b in zip([1] + filters, filters):
        nb_params += (1 + 3**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_b) * filters_b
    for filters_a, filters_b in zip(filters[::-1], filters[::-1][1:]):
        nb_params += (1 + 2**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_b) * filters_b
    nb_params += 1 + filters[0]
    return nb_params


@pytest.mark.parametrize(
    "levels, nb_params",
    [
        (1, nb_params_unet([64])),
        (2, nb_params_unet([64, 128])),
        (3, nb_params_unet([64, 128, 256])),
        (4, nb_params_unet([64, 128, 256, 512])),
        (5, nb_params_unet([64, 128, 256, 512, 1024])),
    ],
)
def test_nb_params_unet(levels: int, nb_params: int):
    model = delta.model.unet(levels=levels)
    assert model.count_params() == nb_params


def test_pixelwise_weighted_binary_crossentropy_seg():
    rng = np.random.default_rng(seed=42)
    shape = (10, 20)
    labels = rng.integers(2, size=shape)
    weights = rng.uniform(size=shape)
    logits = rng.normal(size=shape)
    tf_labels_weights = tf.constant(np.stack([labels, weights], axis=-1))
    tf_logits = tf.constant(logits[:, :, np.newaxis])

    loss = delta.model.pixelwise_weighted_binary_crossentropy_seg(
        tf_labels_weights, tf_logits
    )
    loss_sum = loss.numpy().sum()
    npt.assert_allclose(loss_sum, 8053870.791163433)


def test_pixelwise_weighted_binary_crossentropy_track():
    rng = np.random.default_rng(seed=42)
    shape = (10, 20)
    labels = rng.integers(2, size=shape)
    weights = rng.uniform(size=shape)
    logits = rng.normal(size=shape)
    tf_labels_weights = tf.constant(np.stack([labels, weights], axis=-1))
    tf_logits = tf.constant(logits[:, :, np.newaxis])

    loss = delta.model.pixelwise_weighted_binary_crossentropy_track(
        tf_labels_weights, tf_logits
    )
    loss_sum = loss.numpy().sum()
    npt.assert_allclose(loss_sum, 8053870.791163433)


def test_moma_segmentation_canary():
    config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE
    model = config.model("segmentation")
    path = Path(__file__).parent / "data/images"
    img = delta.utils.read_image(path / "moma_segmentation_canary.png")
    seg = (model.predict(img[np.newaxis, :, :, np.newaxis])[0, :, :, 0] > 0).astype(np.uint8)
    lblnb, _ = cv2.connectedComponents(seg)
    assert lblnb == 6
