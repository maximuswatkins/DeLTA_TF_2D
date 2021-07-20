'''
This script trains the tracking U-Net.

@author: jblugagne
'''
import config as cfg
import os
from model import unet_track
from data import trainGenerator_track
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Files:
training_set = cfg.training_set_track
savefile = cfg.model_file_track

# Training parameters:
batch_size = 5
epochs = 400
steps_per_epoch = 250
patience = 30

#Data generator parameters:
data_gen_args = dict(
    rotation = 1,
    zoom=.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
    timeshiftX=.05,
    timeshiftY=.05,
    timeshift_prob=.1
    )

# Generator init:
myGene = trainGenerator_track(
    batch_size,
    os.path.join(training_set,'img'),
    os.path.join(training_set,'seg'),
    os.path.join(training_set,'previmg'),
    os.path.join(training_set,'segall'),
    os.path.join(training_set,'mot_dau'),
    os.path.join(training_set,'wei'),
    data_gen_args,
    target_size = cfg.target_size_track,
    crop_windows=cfg.crop_windows
    )

# Define model:
model = unet_track(input_size = cfg.target_size_track+(4,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor='loss', verbose=1, save_best_only=True
    )
early_stopping = EarlyStopping(
    monitor='loss', mode='min', verbose=1, patience=patience
    )

# Train:
history = model.fit_generator(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping]
    )