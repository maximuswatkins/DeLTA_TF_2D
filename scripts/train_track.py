"""
This script trains the tracking U-Net.

@author: jblugagne
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import delta
from delta.data import trainGenerator_track
from delta.model import unet_track

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

# Default config
config = delta.config.DEFAULT_CONFIG_2D
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")

# Files:
training_set = config.training_set_path("tracking")
savefile = Path("new_tracking_model.hdf5")

# Training parameters:
batch_size = 2
epochs = 500
steps_per_epoch = 300
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=1,
    zoom=0.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
)

# Generator init:
myGene = trainGenerator_track(
    batch_size,
    training_set / "img",
    training_set / "seg",
    training_set / "previmg",
    training_set / "segall",
    training_set / "mot_dau",
    training_set / "wei",
    data_gen_args,
    target_size=config.target_size_track,
    crop_windows=config.crop_windows,
    shift_cropbox=5,
)


# Define model:
model = unet_track(input_size=config.target_size_track + (4,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="loss", verbose=1, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=patience)

# Train:
history = model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping],
)
