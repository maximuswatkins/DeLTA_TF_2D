<<<<<<< HEAD
"""
This script simply runs the DeLTA pipeline on a movie.

Please download the latest models and evaluation movies first with
delta.assets.download_assets().

In order to use on bioformats-compatible files such .nd2, .czi, .oib, ome-tiff
 etc files please install python-bioformats first.
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:29:10 2021
>>>>>>> 7a3a645d5ad289c57922f7baf2e19060b723df0a

@author: jeanbaptiste
"""

import delta

<<<<<<< HEAD
# Load config ('2D' or 'mothermachine'):
delta.config.load_config(presets='2D')

# Use eval movie as example:
file_path = delta.config.eval_movie

# Init reader (use bioformats=True if working with nd2, czi, ome-tiff etc):
xpreader = delta.utils.xpreader(file_path)
=======
# Init reader:
xpreader = delta.utils.xpreader()
>>>>>>> 7a3a645d5ad289c57922f7baf2e19060b723df0a

# Init pipeline:
xp = delta.pipeline.Pipeline(xpreader)

<<<<<<< HEAD
# Run it (you can specify which positions, which frames to run etc):
xp.process()
=======
# Run it:
xp.process()
>>>>>>> 7a3a645d5ad289c57922f7baf2e19060b723df0a
