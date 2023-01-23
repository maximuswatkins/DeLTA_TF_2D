#!/usr/bin/env python3
"""
Handle asset downloads (model files, training sets, and demonstration movies).

@author: jeanbaptiste
"""

from pathlib import Path
from typing import Any, Dict, Literal

import pooch

PRESETS = Literal["2D", "mothermachine"]

MODEL = Literal["rois", "seg", "track"]

BASE_URL = "https://docs.google.com/uc?export=download&confirm=t&id="

MODEL_REGISTRY = pooch.create(
    path=pooch.os_cache("delta") / "models",
    base_url="",
    registry={
        "unet_momachambers_seg.hdf5": "sha256:00156301fc0c8abd84efa64c18bd418acd57f12503ef17f1c4b7d0c7c2fcbe51",
        "unet_moma_seg.hdf5": "sha256:b9e0985013df96bf6042226b0fe2c8313a4fa6623d166e12402563420ab36dad",
        "unet_moma_track.hdf5": "sha256:ee7cc489bb2ea08e167f6f3c4a5d6b3791d0b080a2a8411b714f71428d8eb6ef",
        "unet_pads_seg.hdf5": "sha256:7b011b1a0df760548a78a9d27202610f015d8d40730a095a0b6c4fe61e1c1eb2",
        "unet_pads_track.hdf5": "sha256:9c93a068fb3f424146f15a539a823107f2932c710e294b92274827f5648bf946",
    },
    urls={
        "unet_momachambers_seg.hdf5": BASE_URL + "13SFG0T73XG8NfVVwLyGRJ_smZdMiWLM_",
        "unet_moma_seg.hdf5": BASE_URL + "1o4Ur0bQ7IvqCkfjeNFYdoEjpAsN7b2uK",
        "unet_moma_track.hdf5": BASE_URL + "1IRzCcP0mjdRRZ4Xzj9qSw4QI958apdTS",
        "unet_pads_seg.hdf5": BASE_URL + "1PSqYtEdJJNaxv7xwDmcZ7X2JomnlF-dP",
        "unet_pads_track.hdf5": BASE_URL + "1EPRV-vF5Dv-UDOn0fdnpThIgjWxNZDio",
    },
    env="DELTA_ASSETS_CACHE",
)

TRAININGSETS_REGISTRY = pooch.create(
    path=pooch.os_cache("delta") / "training_sets",
    base_url="",
    registry={
        "2D.zip": "sha256:dade99b86ab039e842ea97f255003772997d560a1b4610cb969551056512aaf8",
        "mothermachine.zip": "sha256:34dd37a6020049eb4e4db2075dd7dbc42673d3a0f3581d018af65d98c4009e65",
    },
    urls={
        "2D.zip": BASE_URL + "1sWWkIDjIf77KwddHZgX64f8_LxV7hQx_",
        "mothermachine.zip": BASE_URL + "1gzLWospgxqU8oqd8RqZE_WBtQb3N4UJu",
    },
    env="DELTA_ASSETS_CACHE",
)

DEMO_REGISTRY = pooch.create(
    path=pooch.os_cache("delta") / "demo_movies",
    base_url="",
    registry={
        "2D.zip": "sha256:4d14b1121b753e2fe06bd4a15a75fe2941513fc91cda228b60c8aacc028ddfd2",
        "mothermachine.zip": "sha256:ecd0bc8b7e119146a77f3c703c945c9e080e0b7ebe0fcf3a457264888f739de1",
    },
    urls={
        "2D.zip": BASE_URL + "1PpAGwKF5nQnVmwYg2ww5fXPLMLZecah_",
        "mothermachine.zip": BASE_URL + "1cwCLceLucFrKG-YrZ9AjGSrq7Y8z-cHW",
    },
    env="DELTA_ASSETS_CACHE",
)


def download_training_set(presets: PRESETS, model: MODEL) -> Path:
    """
    Return the path of the training set for the currently loaded configuration.

    Depending on the current value of presets ("2D" or "mothermachine") and
    of the value of the argument `model` ("rois", "seg" or "track"), this
    function returns the path of the corresponding training set.  If necessary,
    it downloads it first and caches it under a directory that can be specified
    by the environment variable `DELTA_ASSETS_CACHE`.

    Parameters
    ----------
    presets : PRESETS
        2D or mothermachine
    model : str
        Type of training set: `rois`, `seg` or `track`.

    Returns
    -------
    path : Path
        Path of the training set.

    """
    TRAININGSETS_REGISTRY.fetch(
        f"{presets}.zip", processor=pooch.Unzip(extract_dir=presets), progressbar=True
    )
    path = {
        ("mothermachine", "rois"): "chambers_seg_set/train",
        ("mothermachine", "seg"): "segmentation_set/train_multisets",
        ("mothermachine", "track"): "tracking_set/train_multisets",
        ("2D", "seg"): "segmentation_set",
        ("2D", "track"): "tracking_set",
    }[presets, model]
    return TRAININGSETS_REGISTRY.path / presets / "training" / path


def download_demo_movie(presets: PRESETS) -> Path:
    """
    Return the path of the demo movie for the currently loaded configuration.

    If necessary, it downloads it first and caches it under a directory that
    can be specified by the environment variable `DELTA_ASSETS_CACHE`.

    Parameters
    ----------
    presets : PRESETS
        2D or mothermachine

    Returns
    -------
    path : Path
        Path of the demo movie.
    """
    DEMO_REGISTRY.fetch(
        f"{presets}.zip",
        processor=pooch.Unzip(extract_dir=presets),
        progressbar=True,
    )
    return DEMO_REGISTRY.path / presets


def download_model(presets: PRESETS, model: MODEL) -> Path:
    filename = {
        ("2D", "seg"): "unet_pads_seg.hdf5",
        ("2D", "track"): "unet_pads_track.hdf5",
        ("mothermachine", "rois"): "unet_momachambers_seg.hdf5",
        ("mothermachine", "seg"): "unet_moma_seg.hdf5",
        ("mothermachine", "track"): "unet_moma_track.hdf5",
    }[presets, model]
    return MODEL_REGISTRY.fetch(filename, progressbar=True)


def _ask(prompt: str, options: Dict[str, Any] = {"y": True, "n": False}) -> Any:
    question = f"{prompt} [{'/'.join(options)}]: "
    answer = input(question)
    while answer not in options:
        print(f"invalid answer: '{answer}'")
        answer = input(question)
    return options[answer]
