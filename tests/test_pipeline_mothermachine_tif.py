#!/usr/bin/env python3
"""
Created on Mon Nov  8 10:08:21 2021.

@author: jeanbaptiste
"""

from pathlib import Path

import delta
from delta.config import DEFAULT_CONFIG_MOTHERMACHINE


def test_pipeline_mothermachine():
    xpfolder = Path(__file__).parent / "data/movie_mothermachine_tif"

    xpreader = delta.utils.XPReader(xpfolder)

    xp = delta.pipeline.Pipeline(xpreader, DEFAULT_CONFIG_MOTHERMACHINE)

    xp.process(clear=False)

    pos0 = delta.pipeline.Position.load(
        xpfolder / "expected_results/Position000000.pkl"
    )
    pos1 = delta.pipeline.Position.load(
        xpfolder / "expected_results/Position000001.pkl"
    )

    assert xp.positions[0] == pos0
    assert xp.positions[1] == pos1

    # Testing load from pipeline save:
    # Init new xpreader:
    new_xpreader = delta.utils.XPReader(xpfolder)

    # Load up processor:
    new_xp = delta.pipeline.Pipeline(
        new_xpreader,
        DEFAULT_CONFIG_MOTHERMACHINE,
        reload=True,
        resfolder=xpfolder / "delta_results",
    )

    assert len(xp.positions) == len(new_xp.positions)
