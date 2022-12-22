"""
Module for configuration files and configuration parameters.

@author: jblugagne
"""
import json as _json
import os as _os
import pathlib as _pathlib
import warnings as _warnings
from typing import Optional, Tuple, Union

import tensorflow as tf

_DELTA_DIR = _pathlib.Path(__file__).parent
"DeLTA lib install directory"
_LOADED = None
"Which config file was loaded"

# Parameters:
presets: str = ""
"Type of analysis: can be '2D', 'mothermachine', or your own custom name for presets"
models: Tuple[str, ...] = ()
"Which models will be run"
model_file_rois: str = ""
"Model file for ROIs segmentation"
model_file_seg: str = ""
"Model file for cell segmentation"
model_file_track: str = ""
"Model file for cell tracking"
target_size_rois: Tuple[int, int] = (0, 0)
"ROI U-Net target/input image size"
target_size_seg: Tuple[int, int] = (0, 0)
"segmentation U-Net target/input image size"
target_size_track: Tuple[int, int] = (0, 0)
"tracking U-Net target/input image size"
training_set_rois: str = ""
"Path to ROIs U-Net training set"
training_set_seg: str = ""
"Path to segmentation training set"
training_set_track: str = ""
"Path to tracking training set"
eval_movie: str = ""
"Path to evaluation movie / image sequence"
rotation_correction: bool = False
"Flag to try to automatically correct image rotation (for microfluidic devices)"
drift_correction: bool = False
"Flag to correct drift over time (for microfluidic devices / ROIs)"
crop_windows: bool = False
"Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them"
min_roi_area: float = 0.0
"Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)"
whole_frame_drift: bool = False
"If correcting for drift, use the entire frame instead of the region above the chambers"
min_cell_area: float = 0.0
"Minimum area of detected cells, in pixels. Can be set to 0"
save_format: Tuple[str, ...] = ()
"Format to save output data to"
TF_CPP_MIN_LOG_LEVEL: str = ""
"Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)"
memory_growth_limit: Optional[float] = None
"""If running into OOM issues or having trouble with cuDNN loading, try setting
memory_growth_limit to a value in MB: (eg 1024, 2048...)"""
pipeline_seg_batch: int = 0
"""If running into OOM issues during segmentation with the pipeline, try lowering
this value. You can also try to increase it to improve speed"""
pipeline_track_batch: int = 0
"""If running into OOM issues during tracking with the pipeline, try lowering
this value. You can also try to increase it to improve speed"""

###
# IMPORTANT: Do not change the default parameters below. Update the .json files instead if necessary.
###

_DEFAULTS_2D = dict(
    presets="2D",  # Type of analysis. Can be '2D', 'mothermachine', or your own custom name for presets
    models=("segmentation", "tracking"),  # Which models will be run
    model_file_rois=None,  # Model file for ROIs segmentation (None for 2D)
    model_file_seg=None,  # Model file for cell segmentation (placeholder)
    model_file_track=None,  # Model file for cell tracking(placeholder)
    target_size_rois=(512, 512),  # ROI U-Net target/input image size
    target_size_seg=(512, 512),  # segmentation U-Net target/input image size
    target_size_track=(256, 256),  # tracking U-Net target/input image size
    training_set_rois=None,  # Path to ROIs U-Net training set (None for 2D)
    training_set_seg=None,  # Path to segmentation training set (placeholder)
    training_set_track=None,  # Path to tracking training set (placeholder)
    eval_movie=None,  # Path to evaluation movie / image sequence
    rotation_correction=False,  # Flag to try to automatically correct image rotation (for microfluidic devices)
    drift_correction=False,  # Flag to correct drift over time (for microfluidic devices / ROIs)
    crop_windows=True,  # Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them
    min_roi_area=500,  # Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)
    whole_frame_drift=False,  # If correcting for drift, use the entire frame instead of the region above the chambers
    min_cell_area=20,  # Minimum area of detected cells, in pixels. Can be set to 0
    save_format=("pickle", "legacy", "movie"),  # Format to save output data to.
    TF_CPP_MIN_LOG_LEVEL="2",  # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    memory_growth_limit=None,  # If running into OOM issues or having trouble with cuDNN loading, try setting memory_growth_limit to a value in MB: (eg 1024, 2048...)
    pipeline_seg_batch=64,  # If running into OOM issues during segmentation with the pipeline, try lowering this value. You can also try to increase it to improve speed
    pipeline_track_batch=64,  # If running into OOM issues during tracking with the pipeline, try lowering this value. You can also try to increase it to improve speed
)

_DEFAULTS_mothermachine = dict(
    presets="mothermachine",  # Type of analysis. Can be '2D', 'mothermachine', or your own custom name for presets
    models=("rois", "segmentation", "tracking"),  # Which models will be run
    model_file_rois=None,  # Model file for ROIs segmentation (placeholder)
    model_file_seg=None,  # Model file for cell segmentation (placeholder)
    model_file_track=None,  # Model file for cell tracking(placeholder)
    target_size_rois=(512, 512),  # ROI U-Net target/input image size
    target_size_seg=(256, 32),  # segmentation U-Net target/input image size
    target_size_track=(256, 32),  # tracking U-Net target/input image size
    training_set_rois=None,  # Path to ROIs U-Net training set (None for 2D)
    training_set_seg=None,  # Path to segmentation training set (placeholder)
    training_set_track=None,  # Path to tracking training set (placeholder)
    eval_movie=None,  # Path to evaluation movie / image sequence
    rotation_correction=True,  # Flag to try to automatically correct image rotation (for microfluidic devices)
    drift_correction=True,  # Flag to correct drift over time (for microfluidic devices / ROIs)
    crop_windows=False,  # Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them
    min_roi_area=500,  # Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)
    whole_frame_drift=False,  # If correcting for drift, use the entire frame instead of the region above the chambers
    min_cell_area=20,  # Minimum area of detected cells, in pixels. Can be set to 0
    save_format=("pickle", "legacy", "movie"),  # Format to save output data to.
    TF_CPP_MIN_LOG_LEVEL="2",  # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    memory_growth_limit=None,  # If running into OOM issues or having trouble with cuDNN loading, try setting memory_growth_limit to a value in MB: (eg 1024, 2048...)
    pipeline_seg_batch=64,  # If running into OOM issues during segmentation with the pipeline, try lowering this value. You can also try to increase it to improve speed
    pipeline_track_batch=64,  # If running into OOM issues during tracking with the pipeline, try lowering this value. You can also try to increase it to improve speed
)


def load_config(
    json_file: Optional[Union[str, _pathlib.Path]] = None,
    presets: str = "2D",
    config_level: Optional[str] = None,
):
    """
    Loads json configuration files.

    Parameters
    ----------
    json_file : str or None, optional
        Path to json file containing configuration. If None, load_config will
        search for local or global config files based on the presets value and
        config_level.
        The default is None.
    presets : str, optional
        If json_file is None, search for files named 'config_<presets>.json in
        the local user folder under .delta and then if not found in the global
        folder (ie the install folder).
        The default is '2D'.
    config_level : str or None, optional
        If 'local', look for preset config file in the local user folder under
        ~/.delta. If 'global', look in the delta install folder under
        assets/config. If None, load defaults.
        The default is None.

    Returns
    -------
    None.

    """
    if presets == "2D":
        defaults = _DEFAULTS_2D
    elif presets == "mothermachine":
        defaults = _DEFAULTS_mothermachine
    else:
        raise ValueError(
            """Valid presets are '2D' and 'mothermachine'.
            If you implemented very different presets, please provide
            a config file to load_config() directly
            """
        )

    if json_file is None:
        _local_path = _pathlib.Path.home() / f".delta/config_{presets}.json"
        _global_path = _DELTA_DIR / f".delta/config_{presets}.json"
        if config_level == "local":
            json_file = _local_path
        elif config_level == "global":
            json_file = _global_path
    
    if json_file is None:
        variables = defaults
    else:
        variables = _read_json(_pathlib.Path(json_file))

    # Check if it has the same parameters set as defaults:
    if set(variables.keys()) != set(defaults.keys()):
        _warnings.warn(
            "The config file keys differ from the defaults. This is most "
            "likely because the config file was generated for an earlier version"
            " of DeLTA. This may cause issues, to avoid this either use "
            "delta.assets.download_assets() again or update the config file "
            "manually to match the keys in the defaults. "
            "run print(delta.config._DEFAULTS_2D) "
            "or print(delta.config._DEFAULTS_mothermachine) for defaults."
        )

    # Update config variables:
    globals().update(variables)

    global _LOADED
    _LOADED = json_file

    # Tensorflow technical parameters:
    # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL

    # If running into OOM issues or having trouble with cuDNN loading, try setting
    # memory_growth_limit to a value in MB: (eg 1024, 2048...)
    if memory_growth_limit is not None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_growth_limit
                        )
                    ],
                )
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


def _read_json(json_file: _pathlib.Path):

    # Load file:
    print(f"Loading configuration from: {json_file}")
    with json_file.open("r") as file:
        variables = _json.load(file)

    # Type cast:
    for key, value in variables.items():
        if isinstance(value, list):
            # Always use tuples, not lists in config
            variables[key] = tuple(value)
        elif key.startswith(("model_file", "training_set", "eval_movie")):
            variables[key] = _pathlib.Path(value)

    return variables
