Changelog
=========

All notable changes to this project will be documented in this file,
beginning with version 1.0.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

1.2
---

### Added

-   Added `prep_imdata()` to Tile and Tile subclasses. This method can
    be used to prepare image data before feature detection and matching,
    for example, by converting a 16-bit image to an 8-bit image.

### Changed

-   Fixed bug stitching mosaics that have not been aligned and that have
    inconsistent row heights or column widths. The tiles making up each
    row must still have the same height, and those making up each column
    must have the same width.
-   The `Mosaic.align()` method now raises RuntimeError if it fails to
    detect any features or align at least two tiles.

1.1
---

### Added

-   Added optional channel_order kwarg to the stitch method. Allows
    users to specify the channel order in the stitched array.

1.0
---

### Added

-   Added scikit-image backend in addition to opencv (but it’s very
    slow)
-   Added tests

### Changed

-   **Breaking:** Refactored package and discarded or modified all
    functions and classes
-   Modified functions to make it easier to customize the stitching
    workflow
-   Can now create a mosaic from a grid with unknown structure

### Removed

-   Removed subcommands from command line tool. Use the stitch2d command
    directly to create mosaics instead.
-   Removed ability to set offsets manually using a GUI. Use the limit
    kwarg on the align method and the build_out method to quickly build
    mosaics instead.
-   Removed example tiles
-   Removed modules used to organize tiles generated in the NMNH Mineral
    Sciences labs
