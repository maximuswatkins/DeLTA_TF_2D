# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Changed
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
