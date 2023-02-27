# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- DeLTA now computes the growth rate of the cells.
- It is now possible to specify the number of convolution kernels at each
  layer when creating a U-Net.
- Added support for TensorBoard.
- The function `delta.data.make_weights` can create all of the weight maps
  for a training dataset.
- The function `delta.utilities.read_image` reads and normalises png, tif and
  more.
- The function `load_training_dataset_seg` loads training datasets and allows
  to specify validation and test splits.
- DeLTA's configuration is now its own `Config` object instead of many
  variables dispersed in `delta.config`.  Configuration files are read with
  `Config.read`, and the resulting `config` should be passed to `Pipeline`
  at its creation.
- Positions can now be loaded with the static method `Position.load`.
- Methods were added to the class `CroppingBox`: notably `full` (to create a
  box from a whole reference image) and `crop` (to crop an image according to
  the box).
- Added a commandline interface to DeLTA accessible with `python -m delta`.
### Changed
- All unet models return now logits, instead of probabilities, to improve
  numerical accuracy.
- The function `seg_weights` is faster by a factor 20.
- The function `seg_weights_2D` is faster by a factor 12.
- Upgraded tensorflow dependency to version 2.7.
- Upgraded OpenCV dependency to version 4.6.
- Dropped support for Python 3.7.
- Switched from `setup.py` to `pyproject.toml`.
- DeLTA assets (models, training sets, evaluation movies) are downloaded
  automatically and cached, thanks to the python package `pooch`.
- The function `xpreader.getframes` now only takes one position at a time.
- The functions `extract_poles` and `getpoles` now return a dictionary instead
  of a list.
- The function `getrandomcolors` now returns a dictionary instead of a list.
- The attribute `lineage.cells` is now a dictionary instead of a list.
- `CroppingBox` is now a `dataclass`, instead of a dictionary.
- The function `affine_transform` now takes an `angle` argument to perform a
  rotation.
- The `label_stack` is now created by `process_tracking_outputs`.
- The function `roi_features` now returns a dictionary instead of a list.
- The argument `features` was removed from `extract_features` and
  `roi_features`: now, every possible feature is always returned.
- `python-bioformats` is again installed automatically now.
- The `threshold` argument of `binarizerange` is now compulsory, but one can
  still pass `None` to binarize in the middle of the dynamic range.
- The function `getpoles`'s argument `labels` is now compulsory.
- No need to specify `features` to `Pipeline.process` anymore: they are
  always all returned.
- The functions `track_poles` and `division_poles` (from `utilities`), and
  `createcell`, `updatecell` and `update` (from `Lineage`) changed their
  parameters to now take a `CellFeatures` object instead of a couple of poles.
- The functions `Pipeline.process`, `Position.segment`, `Position.track`,
  `Position.features`, `Position.save`, `Position.results_movie`, and
  `xpreader.getframes` now get an integer argument `nbframes` instead of a list
  `frames`.
- Renamed `xpreader` into `XPReader`.
### Fixed
- Fixed bug in the computation of cell perimeters.
- Fixed training reproducibility bugs by using seeded random generators.
- Fixed edge case bugs in 2D image stitching.
- Fixed edge case bug on pole detection of singular cells.
### Removed
- The function `trainGenerator_seg` was removed.
- The function `download_assets` was removed, because assets are now downloaded
  on the fly.
- The `load_position` was removed in favor of `Position.load`.
- The function `load_config` was removed, replaced by `config.Config.read` and
  `config.Config.load`.
- The function `utils.cropbox` was removed, replaced by `CroppingBox.crop`.
- The function `singlecell_features` was removed because used only once.
- The method `ROI.extract_features` was removed: features are now extracted
  during tracking.
- The old commandline behavior was removed, in favor of `python -m delta`.

## [2.0.5] - 2022-03-11
### Changed
- Adjusted the executable bit of source files
- Removed tifffile from dependency list
- Fixed GPU memory crash when training by using cv2.imread in data module
- Rewrote the `deskew` (detection of rotation) function to be deterministic
  and more accurate
- Corrected Pipeline reloading bug

## [2.0.4] - 2022-02-25
### Changed
- Updated assets module to new google drive download API

## [2.0.3] - 2022-02-02
### Changed
- U-Net "number of levels" now parametrized
- Corrected bugs with poles tracking
- Added poles information to legacy MAT savefiles

## [2.0.2] - 2022-01-17
### Changed
- Rewrote utilities.getrandomcolors to be dependent on cv2 instead of matplotlib

## [2.0.1] - 2022-01-12
### Added
- Added matploblib to requirements.txt

## [2.0.0] - 2022-01-12
### Added
- Full-python pipeline for both 2D and mother machine time-lapse analysis
- [python-bioformats](https://github.com/CellProfiler/python-bioformats) integration
- [Online documentation](https://delta.readthedocs.io/en/latest/)
- [PyPI package](https://pypi.org/project/delta2/)
- [conda-forge package](https://anaconda.org/conda-forge/delta2)
- [Google Colab notebook](https://colab.research.google.com/drive/1UL9oXmcJFRBAm0BMQy_DMKg4VHYGgtxZ)
- Example scripts for data analysis and training & evaluation
- pytest tests suite & systematic testing on dev branch
- Systematic type hinting & standardization of data types
- CI/CD pipeline for PyPI deployment
- Changelog

### Changed
- JSON configuration files
- Automated assets download of [training sets, latest models etc...](https://drive.google.com/drive/u/0/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP)
- [Black](https://black.readthedocs.io/en/stable/) formatting

### Removed
- Matlab-related code
