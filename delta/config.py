"""
Module for configuration files and configuration parameters.

@author: jblugagne
"""
import json as _json
import os as _os
import pathlib as _pathlib
import warnings as _warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

import delta

# DeLTA lib install directory
_DELTA_DIR = _pathlib.Path(__file__).parent


@dataclass
class Config:
    """
    Configuration class containing DeLTA parameters.
    """

    presets: str
    """Type of analysis: can be '2D', 'mothermachine', or your own custom name."""
    models: Tuple[str, ...]
    """Which models need to be loaded."""
    model_file_rois: Optional[_pathlib.Path]
    """Path to the hdf5 model for ROI segmentation."""
    model_file_seg: Optional[_pathlib.Path]
    """Path to the hdf5 model for cell segmentation."""
    model_file_track: Optional[_pathlib.Path]
    """Path to the hdf5 model for cell tracking."""
    target_size_rois: Tuple[int, int]
    """Input size of the ROI segmentation model."""
    target_size_seg: Tuple[int, int]
    """Input size of the cell segmentation model."""
    target_size_track: Tuple[int, int]
    """Input size of the cell tracking model."""
    training_set_rois: Optional[_pathlib.Path]
    """Path to the ROI segmentation training set."""
    training_set_seg: Optional[_pathlib.Path]
    """Path to the cell segmentation training set."""
    training_set_track: Optional[_pathlib.Path]
    """Path to the cell tracking training set."""
    eval_movie: Optional[_pathlib.Path]
    """Path to the evaluation movie."""
    rotation_correction: bool
    """Whether or not to correct for image rotation (mothermachine only)."""
    drift_correction: bool
    """Whether or not to correct for drift over time (mothermachine only)."""
    whole_frame_drift: bool
    """If correcting for drift, use the entire frame instead of the region
    above the chambers."""
    crop_windows: bool
    """If True, crop input images into windows of size target_size_seg for
    segmentation, otherwise resize them."""
    min_roi_area: int
    """Minimum area of detected ROIs in pixels (mothermachine only)."""
    min_cell_area: int
    """Minimum area of detected cells in pixels."""
    save_format: Tuple[str, ...]
    """Format to save output data to."""
    tensorflow_log_level: int
    """Debugging level of tensorflow messages (0: most verbose to 3: quiet)."""
    memory_growth_limit: Optional[int]
    """If running into OOM issues or having troupble with CuDNN loading, try
    setting this to a value in MB: 1024, 2048, etc."""
    pipeline_seg_batch: int
    """If running into OOM issues during segmentation with the pipeline, try
    lowering this value.  You can also increase it to improve speed."""
    pipeline_track_batch: int
    """If running into OOM issues during segmentation with the pipeline, try
    lowering this value.  You can also increase it to improve speed."""

    @staticmethod
    def read(path: Union[str, _pathlib.Path], presets: Optional[str] = None):
        """
        Read a configuration file.

        Parameters
        ----------
        path : Union[str, _pathlib.Path]
            Path to specific configuration file. Keywords "local" and "global"
            can also be used. See online documentation for more information.
        presets : Optional[str], optional
            Name of presets to load config for when using "local" or "global"
            configuration. Values can be "2D", "mothermachine", or a custom
            name if you created one. The default is None.

        Returns
        -------
        Config
            Loaded config object.

        """

        if path in ["local", "global"]:
            if presets is None:
                raise ValueError(
                    "If reading from local or global config files, "
                    "presets are required ('2D' or 'mothermachine' or custom)."
                )
            if path == "local":
                path = _pathlib.Path.home() / f".delta/config_{presets}.json"
            else:
                path = _DELTA_DIR / f"assets/config/config_{presets}.json"

        try:
            return Config(**_read_json(_pathlib.Path(path)))
        except TypeError as err:
            raise ValueError(
                "The config file has too many or is missing some parameters. "
                "This is most likely because the config file was generated for "
                "an earlier version of DeLTA. Please update your config based "
                "on the current one, detailed in `delta.config.Config`."
            ) from err

    def training_set_path(self, model: str) -> _pathlib.Path:
        """
        Returns the path of the training set for the specified model.

        If necessary, we download it first and cache it under a directory
        than can be specified with the environment variable `DELTA_ASSETS_CACHE`.

        Parameters
        ----------
        model : str
            Type of training set: `"rois"`, `"seg"` or `"track"`.

        Returns
        -------
        path : Path
            Path of the training set on the disk.
        """

        config_path = self.__dict__[f"training_set_{model}"]
        if config_path:
            return config_path
        return delta.assets.download_training_set(self.presets, model)

    def demo_movie_path(self) -> _pathlib.Path:
        """
        Returns the path of the demo movie for the specified presets.

        If necessary, we download it first and cache it uder a directory
        than can be specified with the environment variable `DELTA_ASSETS_CACHE`.

        Returns
        -------
        path : Path
            Path of the training set on the disk.
        """

        if self.eval_movie:
            return self.eval_movie
        return delta.assets.download_demo_movie(self.presets)

    def model(self, model: str) -> tf.keras.Model:
        """
        Returns the keras model for the specified preset and model type
        ("rois", "segmentation" or "tracking").

        If the variable "model_file_(rois|seg|track)" is non-empty in the
        configuration, it is assumed to be the filename of the model.  If this
        variable is empty, we download the default model of DeLTA and cache it
        under a directory that can be optionally specified with the environment
        variable `DELTA_ASSETS_CACHE`.

        Parameters
        ----------
        model : str
            Model type ("rois", "segmentation", or "tracking").

        Returns
        -------
        model : tf.keras.Model
            The required model.
        """

        if model == "segmentation":
            model = "seg"
        if model == "tracking":
            model = "track"
        if self.__dict__[f"model_file_{model}"]:
            model_path = self.__dict__[f"model_file_{model}"]
        else:
            model_path = delta.assets.download_model(self.presets, model)
        keras_model = tf.keras.models.load_model(model_path, compile=False)

        # In order to be able to use the old models that had a sigmoid as a final activation
        keras_model.layers[-1].activation = None

        return keras_model

    def apply_tensorflow_config(self):
        # Tensorflow technical parameters:
        # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
        _os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(self.tensorflow_log_level)

        # If running into OOM issues or having trouble with cuDNN loading, try setting
        # memory_growth_limit to a value in MB: (eg 1024, 2048...)
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if self.memory_growth_limit is not None and gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                vdconf = tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=self.memory_growth_limit
                )
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [vdconf]
                )
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                _warnings.warn(
                    f"{len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs"
                )
            except RuntimeError as err:
                # Virtual devices must be set before GPUs have been initialized
                _warnings.warn(f"{err}")


###
# IMPORTANT: Do not change the default parameters below.
# Update the .yml files instead if necessary.
###

DEFAULT_CONFIG_2D = Config(
    presets="2D",
    models=("segmentation", "tracking"),
    model_file_rois=None,
    model_file_seg=None,
    model_file_track=None,
    target_size_rois=(512, 512),
    target_size_seg=(512, 512),
    target_size_track=(256, 256),
    training_set_rois=None,
    training_set_seg=None,
    training_set_track=None,
    eval_movie=None,
    rotation_correction=False,
    drift_correction=False,
    crop_windows=True,
    min_roi_area=500,
    whole_frame_drift=False,
    min_cell_area=20,
    save_format=("pickle", "legacy", "movie"),
    tensorflow_log_level=2,
    memory_growth_limit=None,
    pipeline_seg_batch=64,
    pipeline_track_batch=64,
)

DEFAULT_CONFIG_MOTHERMACHINE = Config(
    presets="mothermachine",
    models=("rois", "segmentation", "tracking"),
    model_file_rois=None,
    model_file_seg=None,
    model_file_track=None,
    target_size_rois=(512, 512),
    target_size_seg=(256, 32),
    target_size_track=(256, 32),
    training_set_rois=None,
    training_set_seg=None,
    training_set_track=None,
    eval_movie=None,
    rotation_correction=True,
    drift_correction=True,
    crop_windows=False,
    min_roi_area=500,
    whole_frame_drift=False,
    min_cell_area=20,
    save_format=("pickle", "legacy", "movie"),
    tensorflow_log_level=2,
    memory_growth_limit=None,
    pipeline_seg_batch=64,
    pipeline_track_batch=64,
)


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
        elif key.startswith(("model_file", "training_set", "eval_movie")) and value:
            variables[key] = _pathlib.Path(value)

    return variables
