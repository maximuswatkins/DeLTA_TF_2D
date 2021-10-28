# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 27 17:26:52 2021

# @author: ooconnor
# """

# import os,sys,glob

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','delta')))

# import utilities as utils
# import config_defaults as cfg
# import pipeline
# import pytest

# def _root_dir():
#     return os.path.dirname(os.path.abspath(__file__))

# def test_pipeline_nd2():
#     sys.argv = [sys.argv[0]]
#     sys.argv.append(os.path.join(_root_dir(),'data/movie_nd2/test.nd2'))
#     sys.argv.append(os.path.join(_root_dir(),'data/movie_nd2/results'))

#     cfg.presets = '2D'
    
#     xpreader = utils.xpreader()

#     xp = pipeline.Pipeline(xpreader)
    
#     xp.process()
    
#     return xp
    
# test_pipeline_nd2()
