"""
Train the tracking U-Net.

@author: jblugagne
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import delta
from delta.data import trainGenerator_track
from delta.model import unet_track
import matplotlib.pyplot as plt
import pickle 

# Set tensorflow's random seed to make the training reproducible
# (only on CPU: not reproducible on GPU yet)
tf.keras.utils.set_random_seed(1)

num_threads = 4

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

# Default config
config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE
# Uncomment if you already have a config file
# config = delta.config.Config.read("/path/to/your/config.yml")
config.apply_tensorflow_config()
# Files:
training_set = config.training_set_path("track")
savefile = Path("new_tracking_model.hdf5")
preset = config.presets

if preset == 'mothermachine':
    preset = 'moma'

# Files:
training_set = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data') / preset
savefile = Path('/projectnb/dunlop/ooconnor/object_detection/cell-trackformer/data') / preset / 'delta' / 'model' / 'moma_track_model.hdf5'
# Training parameters:
batch_size = 2
epochs = 50
steps_per_epoch = None
patience = 50

# Data generator parameters:
data_gen_args = dict(
    rotation=1,
    zoom=0.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
)

train_len = len(list((training_set / 'train' / 'cur_img').glob("*.png")))
val_len = len(list((training_set / 'val' / 'cur_img').glob("*.png")))

steps_per_epoch = train_len // batch_size
validation_steps = val_len // batch_size

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
    ds_train,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping],
)

with open(savefile.parent / 'track_loss_acc_dict.pkl', 'wb') as f:
    pickle.dump(history.history, f)

fig, ax = plt.subplots(1,2)
ax[0].plot([i+1 for i in range(epochs)],history.history['loss'],label='train')
ax[0].plot([i+1 for i in range(epochs)],history.history['val_loss'],label='val')
ax[1].plot([i+1 for i in range(epochs)],history.history['unstack_acc'],label='train')
ax[1].plot([i+1 for i in range(epochs)],history.history['val_unstack_acc'],label='val')

ax[0].set_xlabel('epochs')
ax[1].set_xlabel('epochs')

ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Acc')
ax[1].set_ylim([0,1])

ax[0].legend()
ax[1].legend()
fig.tight_layout()

fig.savefig(savefile.parent / 'track_plot.png')
