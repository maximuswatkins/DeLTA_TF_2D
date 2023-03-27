#!/usr/bin/env python3
"""
Created on Tue Nov  9 16:40:02 2021.

@author: jeanbaptiste
"""
import pytest
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from delta.config import _DELTA_DIR, DEFAULT_CONFIG_2D, DEFAULT_CONFIG_MOTHERMACHINE
from delta.data import load_training_dataset_seg, trainGenerator_track
from delta.model import unet_rois, unet_seg, unet_track


def test_train_moma_rois():
    # Load config
    config = DEFAULT_CONFIG_MOTHERMACHINE
    # Files
    training_set = config.training_set_path("rois")
    savefile = _DELTA_DIR / "assets/test_models/test_train_mothermachine_rois.hdf5"

    # Parameters:
    epochs = 3
    steps_per_epoch = 30
    patience = 50

    # Data generator parameters:
    data_gen_args = dict(
        rotation=3,
        shiftX=0.1,
        shiftY=0.1,
        zoom=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        rotations_90d=False,
        histogram_voodoo=True,
        illumination_voodoo=True,
        gaussian_noise=0.03,
    )

    # Generator init:
    ds_train, ds_val = load_training_dataset_seg(
        dataset_path=training_set,
        target_size=config.target_size_rois,
        crop=False,
        kw_data_aug=data_gen_args,
        validation_split=0.05,
    )

    # Define model:
    model = unet_rois(input_size=config.target_size_rois + (1,))
    model.summary()

    # Callbacks:
    model_checkpoint = ModelCheckpoint(
        savefile, monitor="loss", verbose=1, save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=1, patience=patience
    )

    # Train:
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[model_checkpoint, early_stopping],
    )


@pytest.mark.parametrize("presets, batch_size", [("2D", 1), ("mothermachine", 3)])
def test_train_seg(presets, batch_size):
    # Load config
    if presets == "2D":
        config = DEFAULT_CONFIG_2D
    else:
        config = DEFAULT_CONFIG_MOTHERMACHINE
    # Files
    training_set = config.training_set_path("seg")
    savefile = _DELTA_DIR / f"assets/test_models/test_train_{presets}_seg.hdf5"

    # Training parameters:
    # batch_size = 2
    epochs = 3
    steps_per_epoch = 30
    patience = 50

    # Data generator parameters:
    data_gen_args = dict(
        rotation=2,
        rotations_90d=presets == "2D",
        zoom=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        illumination_voodoo=True,
        gaussian_noise=0.03,
        gaussian_blur=1,
    )

    # Generator init:
    ds_train, ds_val = load_training_dataset_seg(
        dataset_path=training_set,
        target_size=config.target_size_seg,
        crop=presets == "2D",
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
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=2, patience=patience
    )

    # Train:
    model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[model_checkpoint, early_stopping],
    )


@pytest.mark.parametrize("presets", ["2D", "mothermachine"])
def test_train_track(presets):
    # Load config
    if presets == "2D":
        config = DEFAULT_CONFIG_2D
    else:
        config = DEFAULT_CONFIG_MOTHERMACHINE
    # Files
    training_set = config.training_set_path("track")
    savefile = savefile = (
        _DELTA_DIR / f"assets/test_models/test_train_{presets}_track.hdf5"
    )

    # Training parameters:
    batch_size = 2
    epochs = 3
    steps_per_epoch = 30
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
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=1, patience=patience
    )

    # Train:
    model.fit(
        myGene,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping],
    )
