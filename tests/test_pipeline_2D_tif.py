#!/usr/bin/env python3
"""
Created on Wed Oct 20 17:50:34 2021.

@author: ooconnor
"""
import copy
import sys
import tempfile
from pathlib import Path

import pytest

import delta
from delta.config import DEFAULT_CONFIG_2D


@pytest.mark.order(2)
def test_pipeline_tif_stack():
    # 'Pretend' arguments were passed to the cmd line:
    sys.argv = [sys.argv[0]]
    xpfolder = Path(__file__).parent / "data/movie_2D_tif"
    sys.argv.append(str(xpfolder))

    # Init reader
    xpreader = delta.utils.xpreader()

    # Init pipeline:
    xp = delta.pipeline.Pipeline(xpreader, DEFAULT_CONFIG_2D)

    xp.process(frames=[0, 1, 2, 3], clear=False)

    pos0 = delta.pipeline.Position.load(
        xpfolder / "expected_results/Position000000.pkl"
    )

    assert xp.positions[0] == pos0

    # Testing position pickle save

    with tempfile.TemporaryDirectory() as tempdirname:
        assert len(xp.positions) > 0
        pos = xp.positions[0]
        pos.save(filename=tempdirname + "position0", save_format=("pickle",))

        copy.deepcopy(pos)

        # Testing Position.clear
        pos.clear()
        for k in pos.__dict__.keys():
            if k not in pos._pickle_skip:
                assert getattr(pos, k) is None

        # Testing Position.load
        pos = delta.pipeline.Position.load(tempdirname + "position0.pkl")

    # Testing load from pipeline save:
    # Init new xpreader:
    new_xpreader = delta.utils.xpreader(xpfolder)

    # Load up processor:
    new_xp = delta.pipeline.Pipeline(
        new_xpreader,
        DEFAULT_CONFIG_2D,
        reload=True,
        resfolder=xpfolder / "delta_results",
    )

    assert len(xp.positions) == len(new_xp.positions)
