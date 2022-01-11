stitch2d
========

stitch2D is a Python script that stitches a two-dimensional grid of tiles
into a mosaic. It was originally developed for stitching together images
collected on various microscopes in the Department of Mineral Sciences at
the Smithsonian National Museum of Natural History.

When tiles are stitched together by stitch2d, they are translated, not
rotated, resized, or warped. As a result, stitch2d requires all images to
be the same size and orientation. Images must overlap, although they don't
necessarily need to be arranged in a grid.

Source code for this project is located on
`GitHub <https://github.com/adamancer/stitch2d>`_.

Contents
--------

.. toctree::
   :maxdepth: 2

   user-guide
   api
