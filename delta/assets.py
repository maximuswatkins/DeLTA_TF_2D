#!/usr/bin/env python3
"""
Module to handle assets (model files, training sets, evalution movies...),
downloads, and writing configuration files

@author: jeanbaptiste
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests

from .config import _DEFAULTS_2D, _DELTA_DIR, _DEFAULTS_mothermachine

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
    load_models: Optional[Union[bool, str]] = None,
    load_sets: Optional[Union[bool, str]] = None,
    load_evals: Optional[Union[bool, str]] = None,
    config_level: Optional[Union[bool, str]] = None,
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
        If 'local', new config files will be written under the user's local
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
    model_files = _download_models(models_dir=models_dir) if load_models else {}

    # Should we download training sets?
    sets_dir = load_sets if isinstance(load_sets, str) else None
    load_sets = (
        _ask("Download latest training sets?") if load_sets is None else bool(load_sets)
    )
    sets_files = _download_training_sets(sets_dir=sets_dir) if load_sets else {}

    # Should we download evaluation movies?
    movies_dir = load_evals if isinstance(load_evals, str) else None
    load_evals = (
        _ask("Download evaluation movies?") if load_evals is None else bool(load_evals)
    )
    eval_files = _download_eval_movies(movies_dir=movies_dir) if load_evals else {}

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
    config_level: str = "local",
    model_files: Dict[str, str] = {},
    sets_files: Dict[str, str] = {},
    eval_files: Dict[str, str] = {},
) -> None:

    if config_level == "local":
        config_dir = Path.home() / ".delta"
    elif config_level == "global":
        config_dir = _DELTA_DIR / "assets/config"
    elif config_level.exists():
        config_dir = config_level
    else:
        raise ValueError(
            f"""Invalid configs level/path: {config_level}
            If providing a folder path, the folder must exist
            """
        )

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
        config["training_set_rois"] = sets_files["mothermachine"] / "training/chambers_seg_set/train"
        config["training_set_seg"] = sets_files["mothermachine"] / "training/segmentation_set/train_multisets"
        config["training_set_track"] = sets_files["mothermachine"] / "training/tracking_set/train_multisets"
    if len(eval_files) > 0:
        config["eval_movie"] = eval_files["mothermachine"] / "movie_tifs"
    _write_json(config, config_dir / "config_mothermachine.json")

    print(f"Config files written to {config_dir}")


def _download_models(
    model_ids: Dict[str, str] = LATEST_MODELS, models_dir: Optional[Path] = None
) -> Dict[str, str]:
    """
    Download models from gdrive

    Parameters
    ----------
    model_ids : dict, optional
        Dictionary of the filename:google_id for all necessary model files.
        The default is LATEST_MODELS.
    models_dir : Path, optional
        path to save folder. If None, the delta install folder will be used.
        The default is None.

    Returns
    -------
    model_files : dict
        dictionary of save paths for each model.

    """

    if models_dir is None:
        # Retrieve the path of the delta install folder:
        models_dir = _DELTA_DIR / "assets/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be downloaded to {models_dir}")

    model_files = {}
    for filename, google_id in model_ids.items():
        print(f"Downloading model: {filename}...", end="", flush=True)
        model_files[filename] = models_dir / f"{filename}.hdf5"
        _download_file_from_google_drive(google_id, model_files[filename])
        print(" Done")
    return model_files


def _download_training_sets(
    sets_ids: Dict[str, str] = LATEST_TRAININGSETS, sets_dir: Optional[Path] = None
) -> Dict[str, str]:

    if sets_dir is None:
        # Retrieve the path of the delta install folder:
        sets_dir = _DELTA_DIR / "assets/trainingsets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training sets will be downloaded and unzipped to {sets_dir}")

    sets_files = {}
    for filename, google_id in sets_ids.items():
        # Download files:
        sets_files[filename] = sets_dir / filename
        zip_file = sets_dir / f"{filename}.zip"
        _download_and_unzip(
            google_id, zip_file, sets_files[filename], remove_previous=True
        )

    return sets_files


def _download_eval_movies(
    movie_ids: Dict[str, str] = EVAL_MOVIES, movies_dir: Optional[Path] = None
) -> Dict[str, str]:

    if movies_dir is None:
        # Retrieve the path of the delta install folder:
        movies_dir = _DELTA_DIR / "assets/eval_movies"
    movies_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation movies will be downloaded and unzipped to {movies_dir}")

    movie_files = {}
    for filename, google_id in movie_ids.items():
        # Download files:
        movie_files[filename] = movies_dir / filename
        zip_file = movies_dir / f"{filename}.zip"
        _download_and_unzip(
            google_id, zip_file, movie_files[filename], remove_previous=True
        )

    return movie_files


def _download_and_unzip(
    google_id: str, zip_file: str, folder: Path, remove_previous: bool = False
) -> None:

    print(f"Downloading zip: {zip_file}...", end="", flush=True)
    _download_file_from_google_drive(google_id, zip_file)
    print(" Done")

    if remove_previous and folder.exists():
        print("Removing previous extraction...", end="", flush=True)
        shutil.rmtree(folder)
        print(" Done")

    print(f"Extracting to {folder}...", end="", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(zip_file, folder)
    print(" Done")

    print("Deleting archive...", end="", flush=True)
    zip_file.unlink()
    print(" Done")


def _download_file_from_google_drive(google_id: str, destination: Path) -> None:
    # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # Note from JB: the above answer is outdated, google changed the API such
    # that you only need to pass a "t" to confirm, not a token. See merge
    # request 58 for more information

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": google_id, "confirm": "t"}, stream=True)

    _save_response_content(response, destination)


def _save_response_content(response: requests.Response, destination: Path) -> None:
    CHUNK_SIZE = 32768

    with open(destination, "wb") as file:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
