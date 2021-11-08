#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:29:10 2021

@author: jeanbaptiste
"""

import delta

# Init reader:
xpreader = delta.utils.xpreader()

# Init pipeline:
xp = delta.pipeline.Pipeline(xpreader)

# Run it:
xp.process()