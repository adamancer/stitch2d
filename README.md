Stitch2D
========

*Some features in this script require OpenCV 3.0 and ImageMagick.*

A Python script used to stitch a two-dimensional grid of tiles into a mosaic.
You can install it from the command line using pip:

```
pip install stitch2d
```

Users can set offsets manually (if the offset is regular) or automatically
(if OpenCV is installed). If OpenCV is used, tiles that cannot be placed
confidently are excluded from the final mosaic. Tiles are not warped in
either case.

To use, first collect the tilesets you want to stitch as subdirectories
in a single folder. Each subdirectory will be processed using the same
parameters, so offsets for the different tilesets should be similar.
There are three subcommands that can be accessed from the command line.
Information for about of these commands can be accessed from the command
line using -h.

The options available in this module are fairly basic. For more complex
tilesets, consider using the [image stitching plugin](http://fiji.sc/Image_Stitching)
in Fiji.

mosaic
------
Use the mosaic subcommand to stitch together a set of tiles. The resulting
mosaic is saved in the parent of the directory containing the source tiles.
From the command line:

```
stitch2d mosaic
```

That command is perfectly adequate, but you can also specify arguments to
control how your tiles are stitched:

```
stitch2d mosaic -p /path/to/tiles -matcher brute-force \
   -scalar 0.5 -threshold 0.7 --equalize_histogram --create_jpeg
```

Optional arguments include:

*  **-path**: Specifies to path to the source tiles. This argument works in
   all subcommands except organize. If no path is specified, you will be
   prompted to select a directory.
*  **--create_jpeg**: Specifies whether to create a half-size JPEG derivative
   of the final mosaic.
*  **--manual**: Force manual selection of offsets. The script will
   default to manual matching if OpenCV is not installed.

The following arguments can be used to tweak the behavior of OpenCV:

*  **-matcher**: Specifies the algorithm used for feature matching. Must
   be either "brute-force" or "flann"; "brute-force" is the default.
*  **-scalar**: Specifies the amount by which to resize source tiles
   before attempting to match features. Must be a decimal between 0 and 1;
   the default value is 0.5. Smaller values are faster but potentially less
   accurate. The mosaic itself will always use full-size tiles.
*  **-threshold**: The threshold for the Lowe test. Must be a decimal
   between 0 and 1; the default value is 0.7. Lower values give fewer but
   better matches.
*  **--equalize_histogram**: Specifies whether to try to equalize histogram
   in the source image. This can increase contrast and produce better matches,
   but increases computation time.

More information about these values can be found in the [OpenCV-Python
tutorials](https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html).

selector
--------
Use the selector tool to select tiles to exclude from future SEM
element mapping. This tool does the following:

*  Creates a points file for use with Noran System Seven. File contains
   the center point of each tile that was kept from the original grid.
   **The points file has not been tested.**
*  Moves excluded tiles to a directory in the source folder. These tiles
   are automatically reintegrated if the selection script is run again.
*  Produces a list of tiles to skip. The mosaic script uses this list to
   fill in gaps in the mosaic where the excluded tiles were removed.
*  Produces a screenshot showing the final selection grid.

To use the select script:

```
stitch2d select
```

Click the tiles you'd like to remove, or click a darkened tile to reinstate it.
As with the mosaic script, the select command accepts an optional path argument
using the -p flag.

organizer
---------
This command organizes
element maps produces by Noran System Seven into element-specific folders
suitable for mosaicking. It accepts optional arguments for the source and
destination directories:

```
stitch2d organize /path/to/source /path/to/destination
```

Recommended Libraries
=====================

OpenCV
------
[OpenCV](http://www.opencv.org/) is a super useful, basically
open source computer vision library. It's a bit complicated to
install. I found the following tutorials useful:

*  [OS X](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/) (check the comments if you have issues getting the Python bindings
  to show up)
*  [Ubuntu](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)
*  [Windows](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html)

ImageMagick
-----------
The Python Imaging Library will sometimes fail to open TIFFs. When the
mosaic script encounters unreadable TIFFs, it uses [ImageMagick](http://www.imagemagick.org/) to create a usable copy of the
entire tile set. If ImageMagick is not installed, this workaround will
fail and the mosaic will not be created.
