#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:50:34 2021

@author: ooconnor
"""
import os,sys,glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','delta')))

import utilities as utils
import config_defaults as cfg
import pipeline
import pytest

def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))
   
def test_pipeline_tif_stack():
    sys.argv = [sys.argv[0]]
    sys.argv.append(os.path.join(_root_dir(),'data/movie_tif'))
    sys.argv.append(os.path.join(_root_dir(),'data/movie_tif/results'))

    cfg.presets = '2D'
    
    xpreader = utils.xpreader()

    xp = pipeline.Pipeline(xpreader)
    
    xp.process(frames=[0,1,2,3])
    
    return xp
    
test_pipeline_tif_stack()


