"""
This script trains the chambers segmentation U-Net.

@author: jblugagne
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from delta.data import load_training_dataset_seg
from delta.model import unet_rois
from delta.utilities import cfg, tensorboard_callback

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

# Load config:
cfg.load_config(presets="mothermachine")

# Files:
training_set = cfg.training_set_rois
savefile = cfg.model_file_rois

# Parameters:
epochs = 600
steps_per_epoch = 250
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=3,
    shiftX=0.1,
    shiftY=0.1,
    zoom=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    rotations_90d=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
    gaussian_noise=0.03,
)

ds_train, ds_val = load_training_dataset_seg(
    dataset_path=training_set,
    target_size=cfg.target_size_rois,
    crop=False,
    kw_data_aug=data_gen_args,
    validation_split=0.05,
)

# Define model:
model = unet_rois(input_size=cfg.target_size_rois + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="loss", verbose=1, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=patience)

# Train:
history = model.fit(
    ds_train,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_val,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping, tensorboard_callback()],
)
