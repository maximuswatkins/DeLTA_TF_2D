#!/usr/bin/env python3
"""
Created on Tue Nov  9 16:40:02 2021.

@author: jeanbaptiste
"""
import pytest
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from delta.data import trainGenerator_seg, trainGenerator_track
from delta.model import unet_rois, unet_seg, unet_track
from delta.utilities import cfg


@pytest.mark.order(3)
def test_train_moma_rois():
    # Load config:
    cfg.load_config(presets="mothermachine", config_level="global")

    # Files:
    training_set = cfg.training_set_rois
    savefile = cfg._DELTA_DIR / "assets/test_models/test_train_mothermachine_rois.hdf5"

    # Parameters:
    batch_size = 1
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
        rotations_90d=True,
        histogram_voodoo=True,
        illumination_voodoo=True,
        gaussian_noise=0.03,
    )

    # Generator init:
    myGene = trainGenerator_seg(
        batch_size,
        training_set / "img",
        training_set / "seg",
        None,
        augment_params=data_gen_args,
        target_size=cfg.target_size_rois,
    )

    # Define model:
    model = unet_rois(input_size=cfg.target_size_rois + (1,))
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


@pytest.mark.order(2)
def test_train_seg_moma():
    _train_seg("mothermachine", batch_size=3)


@pytest.mark.order(2)
def test_train_seg_2D():
    _train_seg("2D")


def _train_seg(presets, batch_size=1):

    # Load config:
    cfg.load_config(presets=presets, config_level="global")

    # Files:
    training_set = cfg.training_set_seg
    savefile = cfg._DELTA_DIR / f"assets/test_models/test_train_{presets}_seg.hdf5"

    # Training parameters:
    # batch_size = 2
    epochs = 3
    steps_per_epoch = 30
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
    early_stopping = EarlyStopping(
        monitor="loss", mode="min", verbose=2, patience=patience
    )

    # Train:
    model.fit(
        myGene,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint, early_stopping],
    )


@pytest.mark.order(2)
def test_train_track_moma():
    _train_track("mothermachine")


@pytest.mark.order(2)
def test_train_track_2D():
    _train_track("2D")


def _train_track(presets):
    # Load config:
    cfg.load_config(presets="2D", config_level="global")

    # Files:
    training_set = cfg.training_set_track
    savefile = savefile = (
        cfg._DELTA_DIR / f"assets/test_models/test_train_{presets}_track.hdf5"
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
        target_size=cfg.target_size_track,
        crop_windows=cfg.crop_windows,
        shift_cropbox=5,
    )

    # Define model:
    model = unet_track(input_size=cfg.target_size_track + (4,))
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
