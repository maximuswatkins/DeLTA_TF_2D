#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:08:21 2021

@author: jeanbaptiste
"""
import os
import sys

# NOTE: This test script doesn't work quite properly yet because we can not
# switch dynamically between configurations. This will be updated in the
# gdrive feature branch. It works if we manually modify config.py
# with presets='mothermachine' in the current configuration.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import delta

def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))

def test_pipeline_mothermachine():
    sys.argv = [sys.argv[0]]
    sys.argv.append(os.path.join(_root_dir(),'data/movie_mothermachine_tif'))
    sys.argv.append(os.path.join(_root_dir(),'data/movie_mothermachine_tif/results'))
    
    xpreader = delta.utils.xpreader()

    xp = delta.pipeline.Pipeline(xpreader)
    
    xp.process()
