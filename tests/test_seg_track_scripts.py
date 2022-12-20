#!/usr/bin/env python3
"""
Created on Wed Apr  6 09:47:07 2022.

@author: ooconnor
"""

from pathlib import Path

import pytest

import delta

test_folder = Path(__file__).parent


@pytest.mark.order(2)
def test_rois_script():

    # Test images folder:
    inputs_folder = test_folder / "data/movie_mothermachine_tif"

    # Load gloabl moma config:
    delta.config.load_config(presets="mothermachine", config_level="global")

    _run_script(
        test_folder.parent / "scripts/segmentation_rois.py",
        _parameters={
            "inputs_folder": inputs_folder,
        },
    )


@pytest.mark.order(2)
@pytest.mark.parametrize(
    "parameters",
    (
        # 2D seg parameters:
        {"presets": "2D", "inputs_folder": test_folder / "data/movie_2D_tif"},
        # Moma seg parameters:
        {
            "presets": "mothermachine",
            "inputs_folder": test_folder / "data/movie_mothermachine_tif/cropped_rois",
        },
    ),
)
def test_segmentation_script(parameters):

    # Load global config:
    delta.config.load_config(presets=parameters["presets"], config_level="global")

    _run_script(test_folder.parent / "scripts/segmentation.py", _parameters=parameters)


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "parameters",
    (
        # 2D parameters:
        {"presets": "2D", "inputs_folder": test_folder / "data/movie_2D_tif"},
        # Moma parameters:
        {
            "presets": "mothermachine",
            "inputs_folder": test_folder / "data/movie_mothermachine_tif/cropped_rois",
        },
    ),
)
def test_tracking_script(parameters):

    # Load global config:
    delta.config.load_config(presets=parameters["presets"], config_level="global")

    # Adding segmentation folder to the parametrrs dict based on inputs folder:
    parameters["segmentation_folder"] = parameters["inputs_folder"] / "segmentation"

    _run_script(test_folder.parent / "scripts/tracking.py", _parameters=parameters)


def _run_script(_scriptfile, _parameters):
    """Read script & strip parameters code, set parameters, execute script."""
    # Read script, and remove parameters block:
    with open(_scriptfile) as reader:
        _script = _strip_parameters(reader)

    # set parameters:
    for _pname in _parameters:
        exec(f"{_pname} = _parameters[_pname]")

    # Run script:
    exec(_script)


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
