from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("delta2")
except PackageNotFoundError:
    __version__ = None
