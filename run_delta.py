#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:06:23 2021

@author: ooconnor
"""


import utilities as utils
from pipeline import Pipeline

# Init reader:
xpreader = utils.xpreader()

# Init pipeline:
xp = Pipeline(xpreader)

# Run it:
xp.process(frames=[i for i in range(10)])