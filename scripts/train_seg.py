"""
Train the cell segmentation U-Net.

@author: jblugagne
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import delta
from delta.data import load_training_dataset_seg
from delta.model import unet_seg
from delta.utilities import tensorboard_callback
from delta.data import load_training_dataset_seg
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
# Default config
config = delta.config.DEFAULT_CONFIG_MOTHERMACHINE

preset = config.presets
if preset == 'mothermachine':
    preset = 'moma'

# Files:
training_set = Path('/projectnb/dunlop/ooconnor/object_detection/data') / preset / 'delta' / 'segmentation_set'
savefile = Path('/projectnb/dunlop/ooconnor/object_detection') / 'delta' / 'model' / 'moma_seg_model.hdf5'

config.apply_tensorflow_config()

# Training parameters:
epochs = 50
steps_per_epoch = None
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

ds_train = load_training_dataset_seg(
    dataset_path=training_set / 'train',
    target_size=config.target_size_seg,
    crop=config.presets=='2D',
    kw_data_aug=data_gen_args,
    stack=True,
    mode='training'
)

ds_val = load_training_dataset_seg(
    dataset_path=training_set / 'val',
    target_size=config.target_size_seg,
    crop=config.presets=='2D',
    kw_data_aug={},
    stack=True,
    mode='evaluation',
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


with open(savefile.parent / 'seg_loss_acc_dict.pkl', 'wb') as f:
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

fig.savefig(savefile.parent / 'seg_plot.png')
