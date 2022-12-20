"""
This script trains the cell segmentation U-Net.

@author: jblugagne
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from delta.data import trainGenerator_seg
from delta.model import unet_seg
from delta.utilities import cfg, tensorboard_callback

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

# Load config:
cfg.load_config(presets="2D")

# Files:
training_set = cfg.training_set_seg
savefile = cfg.model_file_seg

# Training parameters:
batch_size = 1
epochs = 600
steps_per_epoch = 300
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=2,
    rotations_90d=True,
    zoom=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=True,
    gaussian_noise=0.03,
    gaussian_blur=1,
)

# Generator init:
myGene = trainGenerator_seg(
    batch_size,
    training_set / "img",
    training_set / "seg",
    training_set / "wei",
    augment_params=data_gen_args,
    target_size=cfg.target_size_seg,
    crop_windows=cfg.crop_windows,
)

# Define model:
model = unet_seg(input_size=cfg.target_size_seg + (1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor="loss", verbose=2, save_best_only=True
)
early_stopping = EarlyStopping(monitor="loss", mode="min", verbose=2, patience=patience)

# Train:
history = model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping, tensorboard_callback()],
)
