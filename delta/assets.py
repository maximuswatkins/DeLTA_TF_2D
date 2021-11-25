#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to handle assets (model files, training sets, evalution movies...),
downloads, and writing configuration files

@author: jeanbaptiste
"""
import os
import requests
import json
import shutil
from typing import Union, Dict, Any, Optional

from .config import _DELTA_DIR, _DEFAULTS_2D, _DEFAULTS_mothermachine

LATEST_MODELS = {
    "unet_moma_seg": "1o4Ur0bQ7IvqCkfjeNFYdoEjpAsN7b2uK",
    "unet_moma_track": "1IRzCcP0mjdRRZ4Xzj9qSw4QI958apdTS",
    "unet_pads_seg": "1PSqYtEdJJNaxv7xwDmcZ7X2JomnlF-dP",
    "unet_pads_track": "1EPRV-vF5Dv-UDOn0fdnpThIgjWxNZDio",
    "unet_momachambers_seg": "13SFG0T73XG8NfVVwLyGRJ_smZdMiWLM_",
}

LATEST_TRAININGSETS = {
    "2D": "1sWWkIDjIf77KwddHZgX64f8_LxV7hQx_",
    "mothermachine": "1gzLWospgxqU8oqd8RqZE_WBtQb3N4UJu",
}

EVAL_MOVIES = {
    "2D": "1PpAGwKF5nQnVmwYg2ww5fXPLMLZecah_",
    "mothermachine": "1cwCLceLucFrKG-YrZ9AjGSrq7Y8z-cHW",
}


def download_assets(
    load_models: Union[bool, str] = None,
    load_sets: Union[bool, str] = None,
    load_evals: Union[bool, str] = None,
    config_level: Union[bool, str] = None,
) -> None:
    """
    Download assets such as latest models, training sets etc

    Parameters
    ----------
    load_models : bool, str, or None, optional
        If True, the models will be downloaded from 
        https://drive.google.com/drive/u/0/folders/14EvBCMJIk2LpBCQL-jvsuEwUHNYamyyN
        into an assets folder in the library install path. If a path str is provided,
        the models will be downloaded to the folder it points to. If None, an
        interactive prompt will be used.
        The default is None.
    load_sets : bool, str, or None, optional
        If True, the models will be downloaded and extracted from 
        https://drive.google.com/drive/u/0/folders/1f2t71M-dkxKzdl-xfzHo195SnKAuisMl
        into an assets folder in the library install path. If a path str is provided,
        the models will be downloaded to the folder it points to. If None, an
        interactive prompt will be used.
        The default is None.
    load_evals : bool, str, or None, optional
        If True, the models will be downloaded and extracted from 
        https://drive.google.com/drive/u/0/folders/1f2t71M-dkxKzdl-xfzHo195SnKAuisMl
        into an assets folder in the library install path. If a path str is provided,
        the models will be downloaded to the folder it points to. If None, an
        interactive prompt will be used.
        The default is None.
    config_level : bool, str, or None, optional
        If False, no config file is written. If True, same as 'local'
        If 'local', new config files will be written under the user's local \
        config folder, ~/.delta. If 'global', the config files will be written
        to the install folder, to be shared among all users. If a specific dir
        path is provided, the config files will be written there.
        If None, and interactive prompt will let the user pick.
        The default is None.

    Returns
    -------
    None.

    """

    # Should we download models?
    models_dir = load_models if isinstance(load_models, str) else None
    load_models = (
        _ask("Download latest models?") if load_models is None else bool(load_models)
    )
    model_files = _download_models(models_dir=models_dir) if load_models else dict()

    # Should we download training sets?
    sets_dir = load_sets if isinstance(load_sets, str) else None
    load_sets = (
        _ask("Download latest training sets?") if load_sets is None else bool(load_sets)
    )
    sets_files = _download_training_sets(sets_dir=sets_dir) if load_sets else dict()

    # Should we download evaluation movies?
    movies_dir = load_evals if isinstance(load_evals, str) else None
    load_evals = (
        _ask("Download evaluation movies?") if load_evals is None else bool(load_evals)
    )
    eval_files = _download_eval_movies(movies_dir=movies_dir) if load_evals else dict()

    # Should we write a new config file?
    if isinstance(config_level, bool):
        if config_level:
            config_level = "local"
        else:
            config_level = None
    elif config_level is None:
        config_level = _ask(
            "Write new config_.json files pointing to these new assets?",
            options={"local": "local", "global": "global", "no": None},
        )
    assert isinstance(config_level, str) or config_level is None  # for mypy

    if config_level is not None:
        _write_default_configs(
            model_files=model_files,
            sets_files=sets_files,
            eval_files=eval_files,
            config_level=config_level,
        )


def _ask(prompt: str, options: Dict[str, Any] = {"y": True, "n": False}) -> Any:

    answer = None
    prompt += " ["
    for k in options:
        prompt += k + "/"
    prompt = "".join(list(prompt)[:-1]) + "]: "

    while answer is None:

        answer = input(prompt)

        for k, v in options.items():
            if answer.lower() == k:
                return v

        answer = None
        print("invalid answer: '%s'" % answer)


def _write_default_configs(
    config_level: str = "local",
    model_files: Dict[str, str] = dict(),
    sets_files: Dict[str, str] = dict(),
    eval_files: Dict[str, str] = dict(),
) -> None:

    if config_level == "local":
        config_dir = os.path.expanduser("~/.delta")
    elif config_level == "global":
        config_dir = os.path.join(_DELTA_DIR, "assets", "config")
    elif os.path.exists(config_level):
        config_dir = config_level
    else:
        raise ValueError(
            """Invalid configs level/path: %s
            If providing a folder path, the folder must exist
            """
            % config_level
        )

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Write 2D config:
    config = _DEFAULTS_2D.copy()
    if len(model_files) > 0:
        config["model_file_seg"] = model_files["unet_pads_seg"]
        config["model_file_track"] = model_files["unet_pads_track"]
    if len(sets_files) > 0:
        config["training_set_seg"] = os.path.join(
            sets_files["2D"], "training", "segmentation_set"
        )
        config["training_set_track"] = os.path.join(
            sets_files["2D"], "training", "tracking_set"
        )
    if len(eval_files) > 0:
        config["eval_movie"] = os.path.join(eval_files["2D"], "movie_tifs")
    with open(os.path.join(config_dir, "config_2D.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Write mothermachine config:
    config = _DEFAULTS_mothermachine.copy()
    if len(model_files) > 0:
        config["model_file_rois"] = model_files["unet_momachambers_seg"]
        config["model_file_seg"] = model_files["unet_moma_seg"]
        config["model_file_track"] = model_files["unet_moma_track"]
    if len(sets_files) > 0:
        config["training_set_rois"] = os.path.join(
            sets_files["mothermachine"], "training", "segmentation_set", "train"
        )
        config["training_set_seg"] = os.path.join(
            sets_files["mothermachine"],
            "training",
            "segmentation_set",
            "train_multisets",
        )
        config["training_set_track"] = os.path.join(
            sets_files["mothermachine"], "training", "tracking_set", "train_multisets"
        )
    if len(eval_files) > 0:
        config["eval_movie"] = os.path.join(eval_files["mothermachine"], "movie_tifs")
    with open(os.path.join(config_dir, "config_mothermachine.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("Config files written to %s" % config_dir)


def _download_models(
    model_ids: Dict[str, str] = LATEST_MODELS, models_dir: str = None
) -> Dict[str, str]:
    """
    Download models from gdrive

    Parameters
    ----------
    model_ids : dict, optional
        Dictionary of the filename:google_id for all necessary model files.
        The default is LATEST_MODELS.
    models_dir : None or str, optional
        path to save folder. If None, the delta install folder will be used.
        The default is None.

    Returns
    -------
    model_files : dict
        dictionary of save paths for each model.

    """

    if not isinstance(models_dir, str):
        # Retrieve the path of the delta install folder:
        models_dir = os.path.join(_DELTA_DIR, "assets", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    print("Models will be downloaded to %s" % models_dir)

    model_files = dict()
    for filename, google_id in model_ids.items():
        print("Downloading model: %s..." % filename, end="")
        model_files[filename] = os.path.join(models_dir, filename + ".hdf5")
        _download_file_from_google_drive(google_id, model_files[filename])
        print(" Done")
    return model_files


def _download_training_sets(
    sets_ids: Dict[str, str] = LATEST_TRAININGSETS, sets_dir: str = None
) -> Dict[str, str]:

    if not isinstance(sets_dir, str):
        # Retrieve the path of the delta install folder:
        sets_dir = os.path.join(_DELTA_DIR, "assets", "trainingsets")
    if not os.path.exists(sets_dir):
        os.makedirs(sets_dir)
    print("Training sets will be downloaded and unzipped to %s" % sets_dir)

    sets_files = dict()
    for filename, google_id in sets_ids.items():
        # Download files:
        sets_files[filename] = os.path.join(sets_dir, filename)
        zip_file = os.path.join(sets_dir, filename + ".zip")
        _download_and_unzip(
            google_id, zip_file, sets_files[filename], remove_previous=True
        )

    return sets_files


def _download_eval_movies(
    movie_ids: Dict[str, str] = EVAL_MOVIES, movies_dir: str = None
) -> Dict[str, str]:

    if not isinstance(movies_dir, str):
        # Retrieve the path of the delta install folder:
        movies_dir = os.path.join(_DELTA_DIR, "assets", "eval_movies")
    if not os.path.exists(movies_dir):
        os.makedirs(movies_dir)
    print("Evaluation movies will be downloaded and unzipped to %s" % movies_dir)

    movie_files = dict()
    for filename, google_id in movie_ids.items():
        # Download files:
        movie_files[filename] = os.path.join(movies_dir, filename)
        zip_file = os.path.join(movies_dir, filename + ".zip")
        _download_and_unzip(
            google_id, zip_file, movie_files[filename], remove_previous=True
        )

    return movie_files


def _download_and_unzip(
    id: str, zip_file: str, folder: str, remove_previous: bool = False
) -> None:

    print("Downloading zip: %s" % zip_file)
    _download_file_from_google_drive(id, zip_file)
    print(" Done")

    if remove_previous and os.path.exists(folder):
        print("Removing previous extraction...", end="")
        shutil.rmtree(folder)
        print(" Done")

    print("Extracting to %s..." % (folder), end="")
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.unpack_archive(zip_file, folder)
    print(" Done")

    print("Deleting archive...", end="")
    os.remove(zip_file)
    print(" Done")


def _download_file_from_google_drive(id: str, destination: str) -> None:
    # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response: requests.Response, destination: str) -> None:
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
