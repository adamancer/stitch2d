User guide
==========

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

-   :py:class:`~stitch2d.mosaic.Mosaic`, which incorporates no
    information about how the tiles in the mosaic are arranged
-   :py:class:`~stitch2d.mosaic.StructuredMosaic`, which arranges the
    tiles into a grid based on parameters supplied by the user

You can also use :py:func:`~stitch2d.mosaic.create_mosaic`, as above,
which accepts the same arguments as
:py:class:`~stitch2d.mosaic.StructuredMosaic`. This function returns a
:py:class:`~stitch2d.mosaic.StructuredMosaic` if grid parameters are
provided or can be inferred from the filenames of the tiles or a
:py:class:`~stitch2d.mosaic.Mosaic` if not.

### Mosaic

Since :py:class:`~stitch2d.mosaic.Mosaic` doesn’t know anything about
the tile structure, it can be slow, especially for large grids where
lots of tiles need to be compared. It’s almost always faster to use
:py:class:`~stitch2d.mosaic.StructuredMosaic` where possible.

Initialize a :py:class:`~stitch2d.mosaic.Mosaic` by pointing it to the
directory where the tiles of interest live:

``` python
from stitch2d import Mosaic

mosaic = Mosaic("/path/to/tiles")
```

:py:class:`~stitch2d.mosaic.Mosaic` also includes a class attribute,
:py:attr:`~stitch2d.mosaic.Mosaic.num_cores`, to specify how many cores
it should use when aligning and stitching a mosaic. By default, it uses
one core. Modify this value with:

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
:py:meth:`~stitch2d.mosaic.Mosaic.smooth_seams`, which aligns
brightness/contrast between neighboring tiles by comparing areas of
overlap:

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
:py:attr:`~stitch2d.mosaic.Mosaic.params` attribute, which can be saved
as JSON:

``` python
mosaic.save_params("params.json")
```

Those parameters can then be loaded into a new mosaic if needed:

``` python
mosaic.load_params("params.json")
```

### StructuredMosaic

:py:class:`~stitch2d.mosaic.StructuredMosaic` allows the user to specify
how the tiles in the mosaic should be arranged. For tilesets of known
structure, it is generally faster but otherwise works the same as
:py:class:`~stitch2d.mosaic.Mosaic`. Initialize a structured mosaic
with:

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
acceptable, :py:class:`~stitch2d.mosaic.StructuredMosaic` can use its
knowledge of the tile grid to quickly build a mosaic based on the
positions of only a handful of tiles:

``` python
# Stop aligning once 5 tiles have been successfully placed
mosaic.align(limit=5)

# Build the rest of the mosaic based on the positioned tiles. If from_placed
# is True, missing tiles are appended to the already positioned tiles. If
# False, a new mosaic is calculated from scratch.
mosaic.build_out(from_placed=True)
```

The :py:meth:`~stitch2d.mosaic.StructuredMosaic.build_out` method can
also be used to ensure that all tiles (including those that could not be
placed using feature matching) appear in the final mosaic. The primary
disadvantage of this method is that the placement of those tiles is less
precise.

Beyond 8-bit images
-------------------

**New in 1.2:** The Tile class now includes a
:py:func:`~stitch2d.tile.Tile.prep_imdata` method that can be used to
tweak the image data being used to align the mosaic. When using the
default OpenCVTile class, this method creates an 8-bit copy of the image
data to use for feature detection and matching while retaining the
original data to use when building the mosaic.

The default behavior of :py:func:`~stitch2d.tile.Tile.prep_imdata` is
simplistic. To customize it, use a subclass. For example, the default
method scales the intensities of the original data based on the maximum
intensity found in the array. For images with a small number of
extremely bright pixels, this can yield unusably dim images. A better
approach may be to use `np.percentile()`:

``` python
import numpy as np

class MyTile(OpenCVTile):

    def prep_imdata(self):
        imdata = self.imdata - self.imdata.min()
        return  np.uint8(255 * imdata / np.percentile(imdata, 99))

mosaic = create_mosaic("path/to/tiles", tile_class=MyTile)
```

Similar tools
-------------

The opencv package includes [powerful tools for stitching 2D and 3D
images]((https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html)).
Much of that functionality has been ported to Python as the
[stitching](https://github.com/lukasalexanderweber/stitching) package,
which streamlines the opencv API and includes a useful
[tutorial](https://github.com/lukasalexanderweber/stitching_tutorial). I
didn’t have any luck getting it to work consistently with microscope
tilesets, but it includes advanced features missing from this package
(lens corrections, affine transformations beyond simple translation,
etc.) and can be configured to work with 2D images. It’s definitely
worth a look for tilesets more complex than the simple case handled
here.

[Fiji](https://imagej.net/software/fiji/) also includes a 2D/3D
stitching tool.
