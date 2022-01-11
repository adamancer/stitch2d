Changelog
=========

All notable changes to this project will be documented in this file,
beginning with version 1.0.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

1.0
---

### Added

-   Added scikit-image backend in addition to opencv (but it's very
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
