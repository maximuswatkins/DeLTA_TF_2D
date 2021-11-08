#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:50:34 2021

@author: ooconnor
"""
import os
import sys
import glob

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import delta

def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.mark.order(2)
def test_pipeline_tif_stack():
    
    # 'Pretend' arguments were passed to the cmd line:
    sys.argv = [sys.argv[0]]
    sys.argv.append(os.path.join(_root_dir(),'data/movie_tif'))
    sys.argv.append(os.path.join(_root_dir(),'data/movie_tif/results'))

    # Load configuration:
    delta.utils.cfg.load_config(presets='2D',config_level='global')
    
    # Init reader
    xpreader = delta.utils.xpreader()

    # Init pipeline:
    xp = delta.pipeline.Pipeline(xpreader)
    
    xp.process(frames=[0,1,2,3])
    

    
# test_pipeline_tif_stack()


