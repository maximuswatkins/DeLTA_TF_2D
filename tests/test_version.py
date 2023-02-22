import delta


def test_version():
    version = delta.__version__
    assert isinstance(version, str) and len(version) > 0
