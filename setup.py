import subprocess
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent

VERSION = (
    subprocess.run(["git", "describe"], capture_output=True)
    .stdout.decode("utf-8")
    .strip()
)

PACKAGE_NAME = "delta2"
AUTHOR = "Jean-Baptiste Lugagne, Owen OConnor"
AUTHOR_EMAIL = "jblugagne@bu.edu, ooconnor@bu.edu"
URL = "https://gitlab.com/dunloplab/delta"

LICENSE = "MIT License"
DESCRIPTION = "Segments and tracks bacteria"
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"
URL = "https://gitlab.com/dunloplab/delta"

INSTALL_REQUIRES = [
    "scikit-image>=0.18",
    "opencv-python>=4.5",
    "tensorflow>=2.7",
    "ffmpeg-python>=0.2",
    "requests",
    "numpy>=1.21",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=["delta", "delta.*"]),
    python_requires=">=3.7,<3.11",
    setup_requires=["pytest-order", "pytest-runner", "flake8"],
    tests_require=["pytest"],
    package_data={
        "tests": [
            "data/movie_2D_tif/*.tif",
            "data/movie_mothermachine_tif/*.tif",
            "data/images/*.tif",
        ]
    },
)
