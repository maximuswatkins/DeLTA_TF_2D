"""
Model and loss/metrics functions definitions.

@author: jblugagne
"""
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import datetime
import time
import re 
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dropout,
    Input,
    MaxPool2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops
from tensorflow.keras.callbacks import Callback

# %% Losses/metrics


def pixelwise_weighted_binary_crossentropy_seg(
    seg_weights: tf.Tensor, logits: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.

    The code is adapted from the Keras TF backend.
    (see their github).

    Parameters
    ----------
    seg_weights : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    logits : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    # The weights are passed as part of the seg_weights tensor:
    seg, weights = tf.unstack(seg_weights, 2, axis=-1)

    seg = tf.expand_dims(seg, -1)
    weights = tf.expand_dims(weights, -1)

    # Make background weights be equal to the model's prediction
    weights = tf.where(weights == 0, tf.math.sigmoid(logits), weights)

    entropy = tf.nn.relu(logits) - logits * seg + tf.math.log1p(tf.exp(-tf.abs(logits)))

    loss = K.mean(weights * entropy, axis=-1)

    loss = 1e6 * loss / tf.math.sqrt(tf.math.reduce_sum(weights))

    return loss


def pixelwise_weighted_binary_crossentropy_track(
    seg_weights: tf.Tensor, logits: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.

    The code is adapted from the Keras TF backend.
    (see their github).

    Parameters
    ----------
    seg_weights : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    logits : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    # The weights are passed as part of the seg_weights tensor:
    seg, weights = tf.unstack(seg_weights, 2, axis=-1)

    seg = tf.expand_dims(seg, -1)
    weights = tf.expand_dims(weights, -1)

    entropy = (
        tf.nn.relu(logits)
        - logits * seg
        + math_ops.log1p(math_ops.exp(-tf.abs(logits)))
    )

    loss = K.mean(weights * entropy, axis=-1)

    loss = 1e6 * loss / tf.math.sqrt(tf.math.reduce_sum(weights))

    return loss


def unstack_acc(seg_weights: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    """
    Compute binary accuracy from a stacked tensor with mask and weight map.

    Parameters
    ----------
    seg_weights : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    logits : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Binary prediction accuracy.

    """
    seg, _weights = tf.unstack(seg_weights, 2, axis=-1)

    seg = tf.expand_dims(seg, -1)

    return keras.metrics.binary_accuracy(seg, logits, threshold=0)


def _unet(
    input_layer: tf.Tensor,
    filters: List[int],
    levels: int,
    conv2d_kwargs: Dict[str, str],
    dropout: float = 0.0,
) -> tf.Tensor:
    """
    Create a U-Net recursively.

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


# %% Models
# Generic unet declaration:
def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation: Optional[str] = None,
    output_classes: int = 1,
    dropout: float = 0,
    levels: int = 5,
    filters: Optional[List[int]] = None,
) -> Model:
    """
    Create a U-Net.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, excluding batch size.
        The default is (256,32,1).
    final_activation : None, string or function, optional
        Activation function for the final 2D convolutional layer. see
        keras.activations
        The default is None (no activation: outputs logits).
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
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
    filters: Optional[List[int]] = None,
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
        final_activation=None,
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=pixelwise_weighted_binary_crossentropy_seg,
        metrics=[unstack_acc],
        jit_compile=False,
        run_eagerly=True,
    )

    return model


# Use the following model for tracking and lineage reconstruction:
def unet_track(
    input_size: Tuple[int, int, int] = (256, 32, 4),
    levels: int = 5,
    filters: Optional[List[int]] = None,
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
        final_activation=None,
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=pixelwise_weighted_binary_crossentropy_track,
        metrics=[unstack_acc],
        jit_compile=False,
        run_eagerly=True,
    )

    return model


# Use the following model for segmentation:
def unet_rois(
    input_size: Tuple[int, int, int] = (512, 512, 1),
    levels: int = 5,
    filters: Optional[List[int]] = None,
) -> Model:
    """
    Segmentation U-Net for ROIs.

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
        final_activation=None,
        output_classes=1,
        levels=levels,
        filters=filters,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )

    return model


import matplotlib.pyplot as plt
import pickle
import os

class PlotLearning(Callback):
    def __init__(self, savepath, history_file='training_history.pkl'):
        super().__init__()
        self.history_file = savepath / history_file
        self.savepath = savepath

        # Load existing history if it exists
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                self.history = pickle.load(f)
        else:
            self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    def on_epoch_end(self, epoch, logs=None):
        # Append new epoch data
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('unstack_acc'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_unstack_acc'))

        # Save history to file
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.history, f)

        epochs = range(1, len(self.history['loss']) + 1)

        # Generate and save plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(epochs,self.history['acc'], label='Train Acc')
        ax1.plot(epochs,self.history['val_acc'], label='Val Acc')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper left')
        
        ax2.plot(epochs,self.history['loss'], label='Train Loss')
        ax2.plot(epochs,self.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper left')

        # Save figure to the directory
        fig.savefig(os.path.join(self.savepath, f'Loss.png'))
        plt.close(fig)

    def on_train_end(self, logs=None):
        print("Training complete. Plots saved to directory:", self.savepath)


# Custom Callback to log epoch duration
class EpochTimeLogger(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        total_epoch_time = time.time() - self.epoch_start_time
        duration_str = str(datetime.timedelta(seconds=int(total_epoch_time)))

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                f.write(f'Epoch 1: {duration_str}\n')
        else:
            with open(self.file_path, 'r') as f:
                text = f.read()
                epochs = re.findall(r'Epoch (\d+):', text)

                last_epoch = int(epochs[-1])

            with open(self.file_path, 'a') as f:
                f.write(f'Epoch {last_epoch+1}: {duration_str}\n')

def get_last_epoch(filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            if not lines:
                print("File is empty")
                return 0

            last_epoch = 0
            for line in lines:
                match = re.search(r"^Epoch (\d+):\s+([\d:]+)", line.strip())
                if match:
                    last_epoch = int(match.group(1))

            return last_epoch

    except FileNotFoundError:
        raise NotImplementedError    

class CustomEarlyStopping(Callback):
    def __init__(self, patience, training_history_file):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.training_history_file = training_history_file
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs=None):

        # Load validation loss history at beginning of each epoch (another callback saves the loss info)
        if Path(self.training_history_file).exists():
            with open(self.training_history_file, 'rb') as f:
                loss_history = pickle.load(f)
            self.val_loss_history = loss_history['val_loss']
            self.best = np.min(self.val_loss_history)
            self.wait = len(self.val_loss_history) - (np.argmin(np.array(self.val_loss_history)) + 1)
            self.stopped_epoch = len(self.val_loss_history)
        else:
            self.stopped_epoch = 0
            self.wait = 0
            self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        # another callback already saves the loss data

        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nEpoch {self.stopped_epoch + 1}: early stopping")
