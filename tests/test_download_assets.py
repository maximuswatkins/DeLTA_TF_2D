#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:57:26 2021

@author: jeanbaptiste
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import delta

def test_download_assets():
    
    # Download models and write global config file:
    delta.assets.download_assets(load_models=True, config_level='global')
    
    # Check that all models have been downloaded to the assets folder:
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta'))
    for model_name in delta.assets.LATEST_MODELS:
        filename = os.path.join(assets_dir,'assets','models',model_name+'.hdf5')
        assert os.path.exists(filename), 'Downloaded model file not found: %s'%filename
    
    # Check that config files were created:
    config_file_2d = os.path.join(assets_dir,'assets','config','config_2D.json')
    assert os.path.exists(config_file_2d), '2D config file not found'
    config_file_moma = os.path.join(assets_dir,'assets','config','config_mothermachine.json')
    assert os.path.exists(config_file_moma), 'Mother machine config file not found'
    
    # Check that the config file loads:
    delta.utils.cfg.load_config(config_file_2d)
    assert delta.utils.cfg._LOADED == config_file_2d
    # Check that models can be loaded:
    models = delta.utils.loadmodels()
    assert len(models) > 0
    
    # Check that the config file loads:
    delta.utils.cfg.load_config(config_file_moma)
    assert delta.utils.cfg._LOADED  == config_file_moma
    # Check that models can be loaded:
    models = delta.utils.loadmodels()
    assert len(models) > 0
