stitch2D
========

stitch2D is a Python script that stitches a two-dimensional grid of
tiles into a mosaic. It was originally developed for stitching together
images collected on various microscopes in the Department of Mineral
Sciences at the Smithsonian National Museum of Natural History.

When tiles are stitched together by stitch2d, they are translated, not
rotated, resized, or warped. As a result, stitch2d requires all images
to be the same size and orientation. Images must overlap, although they
don’t necessarily need to be arranged in a grid.

In addition to the instructions below, a guide and API reference are
available in the
[documentation](https://stitch2d.readthedocs.io/en/latest/).

Install
-------

Install stitch2d with pip:

    pip install stitch2d

Or install from the GitHub repository using git and pip:

    git clone https://github.com/adamancer/stitch2d
    cd stitch2d
    pip install .

Quick start
-----------

The following code can be used to create and save a mosaic:

``` python
from stitch2d import create_mosaic


mosaic = create_mosaic("/path/to/tiles")

try:
    mosaic.load_params()
except FileNotFoundError:
    mosaic.downsample(0.6)
    mosaic.align()
    mosaic.reset_tiles()
    mosaic.save_params()

mosaic.smooth_seams()
mosaic.save("mosaic.jpg")
```

A simple stitching workflow is also available from the command line. To
create a smoothed mosaic and save it as a JPEG, run:

    stitch2d path/to/tiles --smooth -output mosaic.jpg

For more information about using this command, including available
parameters, run:

    stitch2d --help

Overview
--------

stitch2d includes two classes that can be used to create mosaics from a
list of tiles:

-   `Mosaic`, which incorporates no information about how the tiles in
    the mosaic are arranged
-   `StructuredMosaic`, which arranges the tiles into a grid based on
    parameters supplied by the user

You can also use `create_mosaic()`, as above, which accepts the same
arguments as `StructuredMosaic`. This function returns a
`StructuredMosaic` if grid parameters are provided or can be inferred
from the filenames of the tiles or a `Mosaic` if not.

### Mosaic

Since `Mosaic` doesn’t know anything about the tile structure, it can be
slow, especially for large grids where lots of tiles need to be
compared. It’s almost always faster to use `StructuredMosaic` where
possible.

Initialize a `Mosaic` by pointing it to the directory where the tiles of
interest live:

``` python
from stitch2d import Mosaic

mosaic = Mosaic("/path/to/tiles")
```

`Mosaic` also includes a class attribute, `num_cores`, to specify how
many cores it should use when aligning and stitching a mosaic. By
default, it uses one core. Modify this value with:

``` python
Mosaic.num_cores = 2
```

Even when using multiple cores, detecting and extracting features can be
time consuming. One way to speed up the process is to reduce the
resolution of the tiles being analyzed:

``` python
mosaic.downsample(0.6)  # downsamples all tiles larger than 0.6 mp
```

Alternatively you can resize the tiles without the size check:

``` python
mosaic.resize(0.6)      # resizes all tiles to 0.6 mp
```

You can then align the smaller tiles:

``` python
mosaic.align()
```

In either case, you can restore the full-size images prior to stitching
the mosaic together:

``` python
mosaic.reset_tiles()
```

Sometimes brightness and contrast can vary significantly between
adjacent tiles, producing a checkerboard effect when the mosaic is
stitched together. This can be mitigated in many cases using
`smooth_seams()`, which aligns brightness/contrast between neighboring
tiles by comparing areas of overlap:

``` python
mosaic.smooth_seams()
```

Once the tiles have been positioned, the mosaic can be viewed:

``` python
mosaic.show()
```

Or saved to a file:

``` python
mosaic.save("mosaic.tif")
```

Or returned as a numpy array if you need more control over the final
mosaic:

``` python
arr = mosaic.stitch()
```

The default backend, opencv, orders color channels as BGR. You may want
to reorder the color channels before working with the image in a
different program. To get an RGB image from a BGR image, use:

``` python
arr = arr[...,::-1].copy()
```

**New in 1.1:** Or specify the desired channel order when stitching:

``` python
arr = mosaic.stitch("RGB")
```

Once the tiles are positioned, their locations are stored in the
`params` attribute, which can be saved as JSON:

``` python
mosaic.save_params("params.json")
```

Those parameters can then be loaded into a new mosaic if needed:

``` python
mosaic.load_params("params.json")
```

### StructuredMosaic

`StructuredMosaic` allows the user to specify how the tiles in the
mosaic should be arranged. For tilesets of known structure, it is
generally faster but otherwise works the same as `Mosaic`. Initialize a
structured mosaic with:

``` python
from stitch2d import StructuredMosaic

mosaic = StructuredMosaic(
    "/path/to/tiles",
    dim=15,                  # number of tiles in primary axis
    origin="upper left",     # position of first tile
    direction="horizontal",  # primary axis (i.e., the direction to traverse first)
    pattern="snake"          # snake or raster
  )
```

For large tilesets where adequate-but-imperfect tile placement is
acceptable, `StructuredMosaic` can use its knowledge of the tile grid to
quickly build a mosaic based on the positions of only a handful of
tiles:

``` python
# Stop aligning once 5 tiles have been successfully placed
mosaic.align(limit=5)

# Build the rest of the mosaic based on the positioned tiles. If from_placed
# is True, missing tiles are appended to the already positioned tiles. If
# False, a new mosaic is calculated from scratch.
mosaic.build_out(from_placed=True)
```

The `build_out()` method can also be used to ensure that all tiles
(including those that could not be placed using feature matching) appear
in the final mosaic. The primary disadvantage of this method is that the
placement of those tiles is less precise.

Similar tools
-------------

The opencv package includes a powerful stitching tool designed for 2D
and 3D images. I didn’t have any luck getting it to work with microscope
tilesets, but it includes advanced features missing from this package
(lens corrections, affine transformations beyond simple translation,
etc.) and can be configured to work with 2D images. It’s definitely
worth a look for tilesets more complex than the simple case handled
here. For code and tutorials, try:

-   [opencv_stitching_tool](https://github.com/opencv/opencv/tree/4.x/apps/opencv_stitching_tool)
-   [opencv_stitching_tutorial](https://github.com/lukasalexanderweber/opencv_stitching_tutorial)
-   [OpenCV: High level stitching API (Stitcher
    class)](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)

[Fiji](https://imagej.net/software/fiji/) also includes a 2D/3D
stitching tool.
