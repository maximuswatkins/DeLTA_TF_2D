"""
This script simply runs the DeLTA pipeline on a movie.

Please download the latest models and evaluation movies first with
delta.assets.download_assets().

In order to use on bioformats-compatible files such .nd2, .czi, .oib, ome-tiff
 etc files please install python-bioformats first.

@author: jeanbaptiste
"""

import delta
from delta.assets import demo_movie_path

# Load config ('2D' or 'mothermachine'):
delta.config._DEFAULT_CONFIG_2D.load()

# Use eval movie as example (or replace by None for interactive selection):
file_path = demo_movie_path()

# Init reader (use bioformats=True if working with nd2, czi, ome-tiff etc):
xpreader = delta.utils.xpreader(file_path)

# Init pipeline:
xp = delta.pipeline.Pipeline(xpreader)

# Run it (you can specify which positions, which frames to run etc):
xp.process()
