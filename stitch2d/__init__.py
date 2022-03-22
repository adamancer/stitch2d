"""Stitches a 2D grid of images into a mosaic"""
from .mosaic import Mosaic, StructuredMosaic, create_mosaic, build_grid, is_grid
from .tile import Tile, OpenCVTile, ScikitImageTile


__version__ = "1.1"
__author__ = "Adam Mansur"
__credits__ = "Smithsonian National Museum of Natural History"
