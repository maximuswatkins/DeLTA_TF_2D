"""
Train the tracking U-Net.

@author: jblugagne
"""
from pathlib import Path
import re
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import delta
from delta.data import trainGenerator_track
from delta.model import unet_track, pixelwise_weighted_binary_crossentropy_track, unstack_acc, EpochTimeLogger, PlotLearning, get_last_epoch, CustomEarlyStopping
from delta.utilities import get_total_time

import numpy as np
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

dataset = 'My2D'

if dataset == 'moma':
    config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE
    epochs = 12
    config.target_size_seg = (256,32)
    config.target_size_track = (256,32)
elif dataset == 'DynamicNuclearNet-tracking-v1_0':
    config = delta.config.DEFAULT_CONFIG_2D
    epochs = 24
    config.target_size_seg = (256,256)
    config.target_size_track = (256,256)
elif dataset == 'My2D':
    config = delta.config.DEFAULT_CONFIG_2D
    epochs = 20
    config.target_size_seg = (256,256)
    config.target_size_track = (256,256)
else:
    raise NotImplementedError

config.number_of_cores = 4
config.apply_tensorflow_config()

# Files:
training_set = Path('/home/max/MaxWork/DeLTA_TF_New/') / 'data' / dataset / 'CTC'
savefile = Path('/home/max/MaxWork/DeLTA_TF_New/') / 'models' / dataset / 'track' / 'my2d_track_model.keras'

RESUME_TRAINING = False

savefile.parents[1].mkdir(exist_ok=True)
savefile.parent.mkdir(exist_ok=True)
log_file = savefile.parent / 'training_time.txt'  # Define the log file path for epoch times

# Training parameters:
batch_size = int(8) #was 1
steps_per_epoch = None
patience = 5

# Data generator parameters:
data_gen_args = dict(
    rotation=1,
    zoom=0.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
)

# Generator init:
ds_train = trainGenerator_track(
    batch_size,
    training_set / "train",
    augment_params = data_gen_args,
    target_size=config.target_size_track,
    crop_windows=config.crop_windows,
    shift_cropbox=5,
)

ds_val = trainGenerator_track(
    batch_size,
    training_set / "val",
    augment_params = {},
    target_size=config.target_size_track,
    crop_windows=config.crop_windows,
    shift_cropbox=0,
)

if RESUME_TRAINING and savefile.exists():
    model = tf.keras.models.load_model(savefile, custom_objects={'pixelwise_weighted_binary_crossentropy_track': pixelwise_weighted_binary_crossentropy_track, 'unstack_acc': unstack_acc})
    last_epoch = get_last_epoch(log_file)
    if last_epoch is None:
       last_epoch = 0
else:
    # Define model:
    model = unet_track(input_size=config.target_size_track + (4,))
    last_epoch = 0

model.summary()

# Callbacks:
training_history_file = savefile.parent / 'training_history.pkl'
custom_early_stopping = CustomEarlyStopping(patience=patience, training_history_file=training_history_file)
model_checkpoint = ModelCheckpoint(savefile, monitor="val_loss", verbose=1, save_best_only=True)

# early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=patience)
epoch_time_logger = EpochTimeLogger(log_file)  # Create instance of the custom callback
plot_learning = PlotLearning(savefile.parent)

regex = re.compile(r'^\d{2}$')
ctc_train_folder_paths = [ctc_folder_path for ctc_folder_path in (training_set / "train").iterdir() if ctc_folder_path.is_dir() and regex.match(ctc_folder_path.name)]

steps_per_epoch_train = 0
for ctc_folder_path in ctc_train_folder_paths:
    steps_per_epoch_train += len(list(ctc_folder_path.glob('*.tif')))
steps_per_epoch_train = int(np.ceil(steps_per_epoch_train / batch_size))

ctc_val_folder_paths = [ctc_folder_path for ctc_folder_path in (training_set / "val").iterdir() if ctc_folder_path.is_dir() and regex.match(ctc_folder_path.name)]

steps_per_epoch_val = 0
for ctc_folder_path in ctc_val_folder_paths:
    steps_per_epoch_val += len(list(ctc_folder_path.glob('*.tif')))
steps_per_epoch_val = int(np.ceil(steps_per_epoch_val / batch_size))

# Checking nan loss error
x, y = next(ds_train)
print("X finite:", np.isfinite(x).all(), "min/max:", x.min(), x.max())
print("Y finite:", np.isfinite(y).all(), "min/max:", y.min(), y.max())

# Train:
model.fit(
    ds_train,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch_train,
    validation_steps=steps_per_epoch_val,
    epochs=epochs-last_epoch,
    callbacks=[model_checkpoint, plot_learning,epoch_time_logger], #,custom_early_stopping
)

# Display and save total time training took
total_time = get_total_time(savefile.parent)
print('Training time {}'.format(total_time))



