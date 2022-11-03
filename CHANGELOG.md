# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- DeLTA now computes the growth rate of the cells
- The function `delta.data.make_weights` can create all of the weight maps
  for a training dataset.
- The function `delta.utilities.read_image` reads and normalises png, tif and
  more.
- The function `load_training_dataset_seg` loads training datasets and allows
  to specify validation and test splits.
### Changed
- Fixed bug in the computation of cell perimeters
- Fixed training reproducibility bugs
- All unet models return now logits, instead of probabilities
- The function `seg_weights` is faster by a factor 20
- The function `seg_weights_2D` is faster by a factor 12
- Upgraded tensorflow dependency to version 2.7
- Upgraded OpenCV dependency to version 4.5
### Removed
- The function `trainGenerator_seg` was removed.

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
- Rewrote utilties.getrandomcolors to be dependent on cv2 instead of matplotlib

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
