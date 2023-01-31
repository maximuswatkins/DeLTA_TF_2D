#!/usr/bin/env python3
"""
Created on Wed Apr  6 09:47:07 2022.

@author: ooconnor
"""

from pathlib import Path

import pytest

from delta.config import _DELTA_DIR, DEFAULT_CONFIG_2D, DEFAULT_CONFIG_MOTHERMACHINE

test_folder = Path(__file__).parent


@pytest.mark.order(2)
def test_rois_script():
    script = test_folder.parent / "scripts/segmentation_rois.py"

    inputs_folder = test_folder / "data/movie_mothermachine_tif"
    parameters = {
        "config": DEFAULT_CONFIG_MOTHERMACHINE,
        "inputs_folder": inputs_folder,
    }
    _run_script(script, _parameters=parameters)


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "config,inputs_folder",
    [
        (DEFAULT_CONFIG_2D, _DELTA_DIR.parent / "tests/data/movie_2D_tif"),
        (
            DEFAULT_CONFIG_MOTHERMACHINE,
            _DELTA_DIR.parent / "tests/data/movie_mothermachine_tif/cropped_rois",
        ),
    ],
)
def test_segmentation_script(config, inputs_folder):
    parameters = {"config": config, "inputs_folder": inputs_folder}
    _run_script(test_folder.parent / "scripts/segmentation.py", _parameters=parameters)


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "config,inputs_folder",
    [
        (DEFAULT_CONFIG_2D, _DELTA_DIR.parent / "tests/data/movie_2D_tif"),
        (
            DEFAULT_CONFIG_MOTHERMACHINE,
            _DELTA_DIR.parent / "tests/data/movie_mothermachine_tif/cropped_rois",
        ),
    ],
)
def test_tracking_script(config, inputs_folder):
    parameters = {"config": config, "inputs_folder": inputs_folder}

    # Adding segmentation folder to the parameters dict based on inputs folder:
    parameters["segmentation_folder"] = parameters["inputs_folder"] / "segmentation"

    _run_script(test_folder.parent / "scripts/tracking.py", _parameters=parameters)


def _run_script(_scriptfile, _parameters):
    """Read script & strip parameters code, set parameters, execute script."""
    # Read script, and remove parameters block:
    with open(_scriptfile) as reader:
        _script = _strip_parameters(reader)

    # set parameters:
    print(_parameters)
    locals().update(_parameters)

    # Run script:
    exec(_script, globals(), locals())


def _strip_parameters(reader):
    """Removes the Parameters /Parameters block from the script."""
    parameters = None
    code = ""
    for line in reader:
        if line.rstrip() == "### Parameters:":
            parameters = True
        elif line.rstrip() == "### /Parameters":
            parameters = False

        if parameters is None or not parameters:
            code += line

    if parameters is None:
        raise RuntimeError('The "### Parameters:" block was not opened in the script')

    if parameters is True:
        raise RuntimeError('The parameters block was not closed with "### /Parameters"')

    return code
