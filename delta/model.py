"""
Module for model definitions and loss/metrics functions definitions

@author: jblugagne
"""
from typing import Callable, Dict, List, Tuple, Union

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Concatenate, Conv2D, Dropout, Input,
                                     MaxPool2D, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import array_ops, math_ops

#%% Losses/metrics


def pixelwise_weighted_binary_crossentropy_seg(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    # Make background weights be equal to the model's prediction
    bool_bkgd = weight == 0 / 255
    weight = tf.where(bool_bkgd, y_pred, weight)

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss


def pixelwise_weighted_binary_crossentropy_track(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss


def unstack_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Unstacks the mask from the weights in the output tensor for
    segmentation and computes binary accuracy

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Binary prediction accuracy.

    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    return keras.metrics.binary_accuracy(seg, y_pred)


def _unet(
    input_layer: tf.Tensor,
    filters: List[int],
    levels: int,
    conv2d_kwargs: Dict[str, str],
    dropout: float = 0.0,
):
    """
    A recursive function to create a U-Net.
    The number of filters at each level are specified with the `filters` argument.

    Parameters
    ----------
    input_layer : tf.Tensor
        The convolutional layer that is the output of the U-Net.
    filters : [int]
        Number of convolutional kernels for each level of the network.
    levels : int
        The total number of layers in the U-Net (to give the correct name to layers).
    conv2d_kwargs : Dict[str, str]
        kwargs for the Conv2D layers of the block.
    dropout : float, default 0.0
        Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
        layer is added.

    Returns
    -------
    output_layer : tf.Tensor
        Output layer of the U-Net (before final convolution).

    """

    name = f"Level{levels - len(filters)}"

    filters_here, *filters_next = filters

    # Initial convolutions
    conv1 = Conv2D(filters_here, 3, **conv2d_kwargs, name=f"{name}_Conv2D_1")(
        input_layer
    )
    conv2 = Conv2D(filters_here, 3, **conv2d_kwargs, name=f"{name}_Conv2D_2")(conv1)

    if dropout > 0.0:
        conv2 = Dropout(dropout, name=f"{name}_Dropout_1")(conv2)

    if not filters_next:
        # This is the last layer
        return conv2

    # Down-sampling
    pool = MaxPool2D(pool_size=(2, 2), name=f"{name}_MaxPooling2D")(conv2)

    # Recursive unet
    output = _unet(
        pool,
        filters=filters_next,
        levels=levels,
        conv2d_kwargs=conv2d_kwargs,
        dropout=dropout,
    )

    # Up-sampling
    up = UpSampling2D(size=(2, 2), name=f"{name}_UpSampling2D")(output)
    conv3 = Conv2D(filters_here, 2, **conv2d_kwargs, name=f"{name}_Conv2D_3")(up)

    # Merge with skip connection layer
    merge = Concatenate(axis=3, name=name + "_Concatenate")([conv2, conv3])

    # Final convolutions
    conv4 = Conv2D(filters_here, 3, **conv2d_kwargs, name=f"{name}_Conv2D_4")(merge)
    conv5 = Conv2D(filters_here, 3, **conv2d_kwargs, name=f"{name}_Conv2D_5")(conv4)

    if dropout > 0.0:
        conv5 = Dropout(dropout, name=f"{name}_Dropout_2")(conv5)

    return conv5


#%% Models
# Generic unet declaration:
def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation: str = "sigmoid",
    output_classes: int = 1,
    dropout: float = 0,
    levels: int = 5,
    filters: List[int] = None,
) -> Model:
    """
    Generic U-Net declaration.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, excluding batch size.
        The default is (256,32,1).
    final_activation : string or function, optional
        Activation function for the final 2D convolutional layer. see
        keras.activations
        The default is 'sigmoid'.
    output_classes : int, optional
        Number of output classes, ie dimensionality of the output space of the
        last 2D convolutional layer.
        The default is 1.
    dropout : float, optional
        Dropout layer rate in the contracting & expanding blocks. Valid range
        is [0,1). If 0, no dropout layer is added.
        The default is 0.
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together. Ignored if `filters` is specified.
        The default is 5.
    filters : [int], optional
        Number of convolutional kernels at each level.  The default is
        starting with 64 and multiplying by 2 at each level.

    Returns
    -------
    model : Model
        Defined U-Net model (not compiled yet).

    """

    # Default conv2d parameters
    conv2d_kwargs = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }

    if filters is None:
        filters = [64 * 2**i for i in range(levels)]
    else:
        levels = len(filters)

    # Inputs layer
    true_input = Input(input_size, name="true_input")

    output = _unet(
        true_input,
        filters=filters,
        levels=levels,
        conv2d_kwargs=conv2d_kwargs,
        dropout=dropout,
    )

    # Final output layer
    true_output = Conv2D(
        output_classes, 1, activation=final_activation, name="true_output"
    )(output)

    model = Model(inputs=true_input, outputs=true_output)

    return model


# Use the following model for segmentation:
def unet_seg(
    pretrained_weights: str = None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
    filters: List[int] = None,
) -> Model:
    """
    Cell segmentation U-Net definition function.

    Parameters
    ----------
    pretrained_weights : hdf5 file, optional
        Model will load weights from hdf5 and start training.
        The default is None
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,1).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together. Ignored if `filters` is specified.
        The default is 5.
    filters : [int], optional
        Number of convolutional kernels at each level.  The default is
        starting with 64 and multiplying by 2 at each level.

    Returns
    -------
    model : Model
        Segmentation U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=pixelwise_weighted_binary_crossentropy_seg,
        metrics=[unstack_acc],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# Use the following model for tracking and lineage reconstruction:
def unet_track(
    pretrained_weights: str = None,
    input_size: Tuple[int, int, int] = (256, 32, 4),
    levels: int = 5,
    filters: List[int] = None,
) -> Model:
    """
    Tracking U-Net definition function.

    Parameters
    ----------
    pretrained_weights : hdf5 file, optional
        Model will load weights from hdf5 and start training.
        The default is None
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,4).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together. Ignored if `filters` is specified.
        The default is 5.
    filters : [int], optional
        Number of convolutional kernels at each level.  The default is
        starting with 64 and multiplying by 2 at each level.

    Returns
    -------
    model : Model
        Tracking U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=pixelwise_weighted_binary_crossentropy_track,
        metrics=[unstack_acc],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# Use the following model for segmentation:
def unet_rois(
    input_size: Tuple[int, int, int] = (512, 512, 1), levels: int = 5, filters: List[int] = None,
) -> Model:
    """
    ROIs segmentation U-Net.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (512,512,1).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together. Ignored if `filters` is specified.
        The default is 5.
    filters : [int], optional
        Number of convolutional kernels at each level.  The default is
        starting with 64 and multiplying by 2 at each level.

    Returns
    -------
    model : Model
        ROIs ID U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")

    return model
