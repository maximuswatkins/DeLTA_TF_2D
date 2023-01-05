#!/usr/bin/env python3
"""
Created on Sun Nov  7 10:57:26 2021.

@author: jeanbaptiste
"""
from pathlib import Path

import pytest

import delta


@pytest.mark.order(1)
def test_download_assets():

    # Download assets and write global config file:
    delta.assets.download_assets(
        load_models=True, load_sets=True, load_evals=True, config_level="global"
    )

    # Check that all models have been downloaded to the assets folder:
    assets_dir = Path(__file__).parents[1] / "delta/assets"
    for model_name in delta.assets.LATEST_MODELS:
        filename = assets_dir / f"models/{model_name}.hdf5"
        assert filename.exists(), f"Downloaded model file not found: {filename}"

    # Check that all sets have been downloaded to the assets folder:
    for set_name in delta.assets.LATEST_TRAININGSETS:
        filename = assets_dir / "trainingsets" / set_name
        assert filename.exists(), f"Extracted training set not found: {filename}"
        zipfile = filename.with_suffix(".zip")
        assert not zipfile.exists(), f"Zip file was not deleted: {zipfile}"

    # Check that all movies have been downloaded to the assets folder:
    for eval_name in delta.assets.EVAL_MOVIES:
        filename = assets_dir / "eval_movies" / eval_name
        assert filename.exists(), f"Extracted evaluation movie not found: {filename}"
        zipfile = filename.with_suffix(".zip")
        assert not zipfile.exists(), f"Zip file was not deleted: {zipfile}"

    # Check that config files were created:
    config_file_2d = assets_dir / "config/config_2D.json"
    assert config_file_2d.exists(), "2D config file not found"
    config_file_moma = assets_dir / "config/config_mothermachine.json"
    assert config_file_moma.exists(), "Mother machine config file not found"

    # Check that the config file loads:
    delta.utils.cfg.load_config(config_file_2d)
    assert delta.utils.cfg._LOADED == config_file_2d
    # Check that models can be loaded:
    models = delta.utils.loadmodels()
    assert len(models) > 0

    # Check that the config file loads:
    delta.utils.cfg.load_config(config_file_moma)
    assert delta.utils.cfg._LOADED == config_file_moma
    # Check that models can be loaded:
    models = delta.utils.loadmodels()
    assert len(models) > 0
