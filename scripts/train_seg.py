"""
Train the cell segmentation U-Net.

@author: jblugagne
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import delta
from delta.model import unet_seg
from delta.utilities import tensorboard_callback
from delta.data import load_training_dataset_seg

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

# Default config
config = delta.config.DEFAULT_CONFIG_2D
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")

# Files:
training_set = config.training_set_path("seg")
savefile = Path("new_segmentation_model.hdf5")

# Training parameters:
epochs = 600
steps_per_epoch = 300
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=2,
    rotations_90d=False,
    zoom=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=True,
    gaussian_noise=0.03,
    gaussian_blur=1,
)

ds_train, ds_val = load_training_dataset_seg(
    dataset_path=training_set,
    target_size=config.target_size_seg,
    crop=config.presets=='2D',
    kw_data_aug=data_gen_args,
    validation_split=0.05,
    stack=True,
)

# Define model:
model = unet_seg(input_size=config.target_size_seg + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="loss", verbose=2, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=2, patience=patience)

# Train:
history = model.fit(
    ds_train,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping, tensorboard_callback()],
)
