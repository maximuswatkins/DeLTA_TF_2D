"""
Train the cell segmentation U-Net.

@author: jblugagne
"""
from pathlib import Path
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import re
import delta
from delta.data_lazy import load_training_dataset_seg
from delta.model import unet_seg, pixelwise_weighted_binary_crossentropy_seg, unstack_acc, EpochTimeLogger, PlotLearning, get_last_epoch, CustomEarlyStopping
from delta.utilities import get_total_time

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
    weights_fcn = delta.data.seg_weights
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
    weights_fcn = delta.data.seg_weights_2D
else:
    raise NotImplementedError

config.number_of_cores = 4
config.apply_tensorflow_config()

# Files:
training_set = Path('/home/max/MaxWork/DeLTA_TF_New/') / 'data' / dataset / 'CTC'
savefile = Path('/home/max/MaxWork/DeLTA_TF_New/') / 'models' / dataset / 'seg' / 'my2d_seg_model.keras'

RESUME_TRAINING = False

savefile.parents[1].mkdir(exist_ok=True)
savefile.parent.mkdir(exist_ok=True)
log_file = savefile.parent / 'training_time.txt'  # Define the log file path for epoch times

config.apply_tensorflow_config()

for folder in ['train','val']:
    if not (training_set / 'delta_weights' / folder / 'wei_seg').exists():
        delta.data.make_weights(training_set / folder,weights_fcn, classweights = (1,1))

# Training parameters:
steps_per_epoch = None
patience = 5

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

if RESUME_TRAINING and savefile.exists():
    model = tf.keras.models.load_model(savefile, custom_objects={'pixelwise_weighted_binary_crossentropy_seg': pixelwise_weighted_binary_crossentropy_seg, 'unstack_acc': unstack_acc})
    last_epoch = get_last_epoch(log_file)
    if last_epoch is None:
       last_epoch = 0
else:
    # Define model:
    model = unet_seg(input_size=config.target_size_seg + (1,))
    last_epoch = 0

model.summary()

# Callbacks:
training_history_file = savefile.parent / 'training_history.pkl'
custom_early_stopping = CustomEarlyStopping(patience=patience, training_history_file=training_history_file)
model_checkpoint = ModelCheckpoint(savefile, monitor="val_loss", verbose=2, save_best_only=True)
epoch_time_logger = EpochTimeLogger(log_file)  # Create instance of the custom callback
plot_learning = PlotLearning(savefile.parent)

# Train:
model.fit(
    ds_train,
    validation_data=ds_val,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs-last_epoch,
    callbacks=[model_checkpoint, epoch_time_logger, plot_learning], #, custom_early_stopping
)

# Display and save total time training took
total_time = get_total_time(savefile.parent)
print('Training time {}'.format(total_time))

