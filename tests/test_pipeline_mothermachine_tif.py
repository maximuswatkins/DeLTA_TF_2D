#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:08:21 2021

@author: jeanbaptiste
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import delta


def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.mark.order(2)
def test_pipeline_mothermachine():
    sys.argv = [sys.argv[0]]
    sys.argv.append(os.path.join(_root_dir(), "data/movie_mothermachine_tif"))
    sys.argv.append(os.path.join(_root_dir(), "data/movie_mothermachine_tif/results"))

    # Load configuration:
    delta.utils.cfg.load_config(presets="mothermachine", config_level="global")

    xpreader = delta.utils.xpreader()

    xp = delta.pipeline.Pipeline(xpreader)

    xp.process()
