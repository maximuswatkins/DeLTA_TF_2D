"""
This script simply runs the DeLTA pipeline on a demo movie.

In order to use on bioformats-compatible files such .nd2, .czi, .oib, ome-tiff
 etc files please install python-bioformats first.

@author: jeanbaptiste
"""

import delta

# Load config ('2D' or 'MOTHERMACHINE'):
delta.config.DEFAULT_CONFIG_2D.load()

# Use demo movie as example (or replace by a path to your own movie):
file_path = delta.assets.demo_movie_path()

# Init reader (use bioformats=True if working with nd2, czi, ome-tiff etc):
xpreader = delta.utils.xpreader(file_path)

# Init pipeline:
xp = delta.pipeline.Pipeline(xpreader)

# Run it (you can specify which positions, which frames to run etc):
xp.process()
