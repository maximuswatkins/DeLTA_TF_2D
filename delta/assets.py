#!/usr/bin/env python3
"""
Module to handle assets (model files, training sets, evaluation movies...),
downloads, and writing configuration files.

@author: jeanbaptiste
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pooch
import tensorflow as tf

import delta.config as cfg

from delta.config import _DEFAULTS_2D, _DELTA_DIR, _DEFAULTS_mothermachine

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


def training_set_path(model: str) -> Path:
    """
    Returns the path of the training set for the currently loaded configuration.

    Depending on the current value of cfg.presets ("2D" or "mothermachine") and
    of the value of the argument `model` ("rois", "seg" or "track"), this
    function returns the path of the corresponding training set.  If necessary,
    it downloads it first and caches it under a directory that can be specified
    by the environment variable `DELTA_ASSETS_CACHE`.

    Parameters
    ----------
    model : str
        Type of training set: `rois`, `seg` or `track`.

    Returns
    -------
    path : Path
        Path of the training set.

    """
    config_path = getattr(cfg, f"training_set_{model}", None)
    if config_path:
        return Path(config_path)
    TRAININGSETS_REGISTRY.fetch(f"{cfg.presets}.zip", processor=pooch.Unzip(extract_dir=cfg.presets))
    path = {
        ("mothermachine", "rois"): "chambers_seg_set/train",
        ("mothermachine", "seg"): "segmentation_set/train_multisets",
        ("mothermachine", "track"): "tracking_set/train_multisets",
        ("2D", "seg"): "segmentation_set",
        ("2D", "track"): "tracking_set",
    }[cfg.presets, model]
    return TRAININGSETS_REGISTRY.path / cfg.presets / "training" / path


def demo_movie_path() -> Path:
    """
    Returns the path of the demo movie for the currently loaded configuration.

    If necessary, it downloads it first and caches it under a directory that
    can be specified by the environment variable `DELTA_ASSETS_CACHE`.

    Returns
    -------
    path : Path
        Path of the demo movie.
    """

    if cfg.eval_movie:
        return Path(cfg.eval_movie)
    DEMO_REGISTRY.fetch(f"{cfg.presets}.zip", processor=pooch.Unzip(extract_dir=cfg.presets))
    return DEMO_REGISTRY.path / cfg.presets


def loadmodels(toload: Optional[Tuple[str, ...]] = None) -> Dict[str, Any]:
    """
    Load models (as specified in config.py)

    If necessary, it downloads it first and caches it under a directory that
    can be specified by the environment variable `DELTA_ASSETS_CACHE`.

    Parameters
    ----------
    toload : tuple of str, or None, optional
        Which of the 3 models to load. If None, cfg.models will be used.
        The default is None.

    Returns
    -------
    models : dict
        Dictionary containing the models specified.

    """

    if toload is None:
        toload = cfg.models

    models = {}

    if "rois" in toload:
        if cfg.model_file_rois:
            model_file_rois = cfg.model_file_rois
        else:
            model_file_rois = MODEL_REGISTRY.fetch(
                cfg.model_file_rois if cfg.model_file_rois else "unet_momachambers_seg.hdf5"
            )
        models["rois"] = tf.keras.models.load_model(model_file_rois, compile=False)

    if "segmentation" in toload:
        if cfg.model_file_seg:
            model_file_seg = cfg.model_file_seg
        else:
            term = {"mothermachine": "moma", "2D": "pads"}[cfg.presets]
            model_file_seg = MODEL_REGISTRY.fetch(f"unet_{term}_seg.hdf5")
        models["segmentation"] = tf.keras.models.load_model(model_file_seg, compile=False)

    if "tracking" in toload:
        if cfg.model_file_track:
            model_file_track = cfg.model_file_track
        else:
            term = {"mothermachine": "moma", "2D": "pads"}[cfg.presets]
            model_file_track = MODEL_REGISTRY.fetch(f"unet_{term}_track.hdf5")
        models["tracking"] = tf.keras.models.load_model(model_file_track, compile=False)

    # In order to be able to use the old models that had a sigmoid as final activation
    for model in models.values():
        model.layers[-1].activation = None

    return models


def _write_json(obj: Any, path: Path) -> None:
    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with path.open(mode="w", encoding="utf8") as file:
        json.dump(obj, file, indent=4, cls=PathEncoder)


def _ask(prompt: str, options: Dict[str, Any] = {"y": True, "n": False}) -> Any:
    question = f"{prompt} [{'/'.join(options)}]: "
    answer = input(question)
    while answer not in options:
        print(f"invalid answer: '{answer}'")
        answer = input(question)
    return options[answer]


def _write_default_configs(
    config_level: Union[str, Path] = "local",
    model_files: Dict[str, Path] = {},
    sets_files: Dict[str, Path] = {},
    eval_files: Dict[str, Path] = {},
) -> None:

    if config_level == "local":
        config_dir = Path.home() / ".delta"
    elif config_level == "global":
        config_dir = _DELTA_DIR / "assets/config"
    else:
        path_config = Path(config_level)
        if not path_config.is_dir():
            raise ValueError(
                f"Invalid config level/path: {config_level}. "
                "If providing a folder path, the folder must exist."
            )
        config_dir = path_config

    config_dir.mkdir(parents=True, exist_ok=True)

    # Write 2D config:
    config = _DEFAULTS_2D.copy()
    if len(model_files) > 0:
        config["model_file_seg"] = model_files["unet_pads_seg"]
        config["model_file_track"] = model_files["unet_pads_track"]
    if len(sets_files) > 0:
        config["training_set_seg"] = sets_files["2D"] / "training/segmentation_set"
        config["training_set_track"] = sets_files["2D"] / "training/tracking_set"
    if len(eval_files) > 0:
        config["eval_movie"] = eval_files["2D"] / "movie_tifs"
    _write_json(config, config_dir / "config_2D.json")

    # Write mothermachine config:
    config = _DEFAULTS_mothermachine.copy()
    if len(model_files) > 0:
        config["model_file_rois"] = model_files["unet_momachambers_seg"]
        config["model_file_seg"] = model_files["unet_moma_seg"]
        config["model_file_track"] = model_files["unet_moma_track"]
    if len(sets_files) > 0:
        config["training_set_rois"] = (
            sets_files["mothermachine"] / "training/chambers_seg_set/train"
        )
        config["training_set_seg"] = (
            sets_files["mothermachine"] / "training/segmentation_set/train_multisets"
        )
        config["training_set_track"] = (
            sets_files["mothermachine"] / "training/tracking_set/train_multisets"
        )
    if len(eval_files) > 0:
        config["eval_movie"] = eval_files["mothermachine"] / "movie_tifs"
    _write_json(config, config_dir / "config_mothermachine.json")

    print(f"Config files written to {config_dir}")

