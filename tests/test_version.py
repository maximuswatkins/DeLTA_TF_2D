import delta


def test_version():
    version = delta.__version__
    assert int(version.split(".")[0]) >= 2
