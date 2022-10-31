import os
from setuptools import setup, find_packages


long_description = (
    "stitch2D is a Python script that stitches a two-dimensional grid of tiles"
    " into a mosaic. It was originally developed for stitching together images"
    " collected on various microscopes in the Department of Mineral Sciences at"
    " the Smithsonian National Museum of Natural History."
    "\n\n"
    " Install with:"
    "\n\n"
    "```\n"
    "pip install stitch2d\n"
    "```"
    "\n\n"
    " When tiles are stitched together by stitch2d, they are translated, not"
    " rotated, resized, or warped. As a result, stitch2d requires all images to"
    " be the same size and orientation. Images must overlap, although they don't"
    " necessarily need to be arranged in a grid."
    "\n\n"
    "Learn more:\n\n"
    "+ [GitHub repsository](https://github.com/adamancer/stitch2d)\n"
    "+ [Documentation](https://stitch2d.readthedocs.io/en/latest/)"
)


setup(
    name="stitch2d",
    maintainer="Adam Mansur",
    maintainer_email="mansura@si.edu",
    description="Stitches a 2D grid of images into a mosaic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.2",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Graphics",
    ],
    url="https://github.com/adamancer/stitch2d.git",
    license="MIT",
    packages=find_packages(),
    install_requires=["joblib", "numpy", "opencv-python", "scikit-image"],
    include_package_data=True,
    entry_points={"console_scripts": ["stitch2d = stitch2d.__main__:main"]},
    zip_safe=False,
)
