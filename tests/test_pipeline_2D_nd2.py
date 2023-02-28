from pathlib import Path

import delta
from delta.config import DEFAULT_CONFIG_2D


def test_pipeline_nd2():
    base_path = Path(__file__).parent / "data/movie_nd2"

    xpreader = delta.utils.XPReader(base_path / "test.nd2")

    xp = delta.pipeline.Pipeline(xpreader, DEFAULT_CONFIG_2D)

    xp.process(clear=False)

    xpreader.close()

    pos0 = delta.pipeline.Position.load(
        base_path / "test_expected_results/Position000000.pkl"
    )

    assert xp.positions[0] == pos0
