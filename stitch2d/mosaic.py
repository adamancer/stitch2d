"""Reads and stitches images from a 2D grid into a mosaic"""
from collections import defaultdict
from itertools import chain
import glob
import imghdr
import json
import logging
import os
import re

from joblib import Parallel, delayed
import numpy as np

from .tile import Tile, OpenCVTile


logger = logging.getLogger(__name__)


class Mosaic:
    """Stitches a mosaic from a list of tiles

    Attributes
    ----------
    grid : list
        tiles arranged into a grid
    shape : tuple
        shape of the mosaic as (height, width[, channels])
    size : tuple
        number of tiles in the mosaic
    tile_class : class
        class to use for tiles in the mosaic
    """

    #: int : Number of cores to use when processing images
    num_cores = 1

    def __init__(self, path_or_tiles, tile_class=None):
        """Initializes a mosaic from a list of tiles

        Parameters
        ----------
        path_or_tiles : str or list-like
            either the path to a directory containing tiles, a list of Tiles,
            or a list of strings or arrays that can be used to create a Tile
        tile_class : class
            class to use for tiles in the mosaic. Defaults to OpenCVTile.
        """

        # Use class of passed Tiles for tile_class
        self.tile_class = tile_class
        if not isinstance(path_or_tiles, str) and not isinstance(path_or_tiles[0], str):

            # Get tile class from 1D list of Tile
            if isinstance(path_or_tiles[0], Tile):
                tile_class = path_or_tiles[0].__class__

            # Get tile class from 2D grid of Tile
            else:
                tile_class = path_or_tiles[0][0].__class__

            if self.tile_class and tile_class != self.tile_class:
                logger.warning(
                    "`tile_class` does not match class of tiles in"
                    " `path_or_tiles`. Using class of tiles instead."
                )

            if not issubclass(tile_class, Tile):
                raise ValueError(
                    f"tile_class must be Tile or subclass if inferred (got {tile_class}"
                )

            self.tile_class = tile_class

        if not self.tile_class:
            self.tile_class = OpenCVTile

        self.shape = None
        self.size = None
        self.grid = None

        self._pool = None

        # Construct the grid underlying the mosaic
        self._build_grid(path_or_tiles)

        logger.info(f"Initialized {self}")

    def __str__(self):
        return f"<{self.__class__.__name__} shape={self.shape}>"

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.tile_class.__class__ == other.tile_class.__class__
            and self.shape == other.shape
            and self.size == other.size
            and self.tiles == other.tiles
            and self.grid == other.grid
        )

    @property
    def tiles(self):
        """Gets a flattened list of all tiles in grid order"""
        return list(chain.from_iterable(self.grid))

    @property
    def placed(self):
        """Calculates number of tiles that have been placed in the mosaic"""
        return len([t for t in self.tiles if t.placed])

    @property
    def params(self):
        """Summarizes parameters needed to stitch mosaic"""

        self._normalize_coordinates()

        # Store coordinates for all placed tiles at full scale
        coords = {}
        filenames = {}
        for i, tile in enumerate(self.tiles):
            if tile.placed:
                coords[i] = [tile.y / tile.scale, tile.x / tile.scale]
            if isinstance(tile.source, str):
                filenames[i] = os.path.basename(tile.source)

        params = {
            "metadata": {
                "shape": list(self.shape),
                "size": self.size,
                "tile_shape": list(self.tiles[0].copy().reset().shape[:2]),
            },
            "coords": coords,
        }
        if filenames:
            params["filenames"] = filenames
        return params

    @property
    def detector(self):
        """Gets a copy of the detector used to align tiles in the mosaic"""
        return self.tiles[0].detector

    @detector.setter
    def detector(self, val):
        for tile in self.tiles:
            tile.detector = val

    @property
    def matcher(self):
        """Gets a copy of the matcher used to align tiles in the mosaic

        Only defined if using OpenCV.
        """
        return self.tiles[0].matcher

    @matcher.setter
    def matcher(self, val):
        for tile in self.tiles:
            tile.matcher = val

    @property
    def pool(self):
        """Returns a shared joblib pool, creating it if needed"""
        if self._pool is None:
            self._pool = Parallel(n_jobs=self.num_cores)
        return self._pool

    @pool.setter
    def pool(self, val):
        self._pool = val

    def placeholder(self, tile=None, fill_value=0):
        """Creates a placeholder tile to fill in gaps in the mosaic

        Arguments
        ---------
        tile : Tile
            tile to base the placeholder on. If not given, uses the first
            tile in the tiles property.
        fill_value : float or int
            fill value

        Returns
        -------
        Tile
            placeholder tile filled with provided value
        """
        if tile is None:
            tile = self.tiles[0]

        imdata = np.full(tile.shape, fill_value, dtype=tile.dtype)
        tile = self.tile_class(imdata)
        tile.is_placeholder = True
        return tile

    def bounds(self):
        """Calculates bounds of the mosaic comprising the placed tiles

        Returns
        -------
        tuple
            bounds of tile as (y1, x1, y2, x2)
        """
        ys = []
        xs = []
        for tile in self.tiles:
            if tile.placed:
                ys.append(tile.y)
                xs.append(tile.x)

        tile = self.tiles[0]
        return min(ys), min(xs), max(ys) + tile.height, max(xs) + tile.width

    def copy(self):
        """Creates a copy of the mosaic based on the grid

        Using the grid instead of the list of tiles allows the grid-building
        step to be skipped when the copy is initialized.

        Returns
        -------
        Mosaic
            copy of the mosaic
        """
        return self.__class__([[t.copy() for t in row] for row in self.grid])

    def save_params(self, path="params.json"):
        """Saves coordinates for placed tiles

        Parameters
        ----------
        path : str
            path to the JSON file
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

        logger.info(f"Saved params for {self} to {path}")

    def load_params(self, path_or_obj="params.json"):
        """Loads coordinates for placed tiles at full scale

        Parameters
        ----------
        path_or_obj : str or dict
            path to the JSON file or param dict from another mosaic

        Raises
        ------
        FileNotFoundError
            thrown if input is path and path not found
        ValueError
            thrown if JSON can't be decoded or does not match this mosaic
        """

        if isinstance(path_or_obj, str):
            try:
                with open(path_or_obj, "r", encoding="utf-8") as f:
                    path_or_obj = json.load(f)
            except (FileNotFoundError, ValueError) as err:
                logger.error(str(err), exc_info=err)
                raise

        # Confirm that metadata matches
        for key, val in self.params["metadata"].items():
            if path_or_obj.get("metadata", {}).get(key) != val:
                msg = (
                    f"JSON param '{key}' does not match this mosaic"
                    f" ({val} != {path_or_obj.get('metadata', {}).get(key)})"
                )
                logger.error(msg)
                raise ValueError(msg)

        # Mosaics are similar, so update tile coordinates from params
        for tile in self.tiles:
            tile.y = None
            tile.x = None

        for key, (y, x) in path_or_obj["coords"].items():
            tile = self.tiles[int(key)]
            tile.y = y * tile.scale
            tile.x = x * tile.scale

        logger.info(f"Loaded previously calculated params for {self}")

    def reset_tiles(self):
        """Reloads tiles at their full resolution"""
        self._normalize_coordinates()
        return self._batch_tile_method("reset")

    def resize(self, size_or_shape, *args, **kwargs):
        """Rescales all tiles in the mosaic using size or shape

        Parameters
        ----------
        size_or_shape : int or tuple of ints
            size in megapixels or shape of resized image
        *args :
            any argument accepted by the resize function used by the tile class
        **kwargs :
            any keyword argument accepted by the resize function used by the
            tile class
        """
        return self._batch_tile_method("resize", size_or_shape, *args, **kwargs)

    def downsample(self, size_or_shape, *args, **kwargs):
        """Downsamples all tiles in the mosaic using the given size or shape

        Parameters
        ----------
        size_or_shape : int or tuple of ints
            size in megapixels or shape of resized image
        *args :
            any argument accepted by the resize function used by the tile class
        **kwargs :
            any keyword argument accepted by the resize function used by the
            tile class
        """
        return self._batch_tile_method("downsample", size_or_shape, *args, **kwargs)

    def detect_and_extract(self):
        """Detects and extracts features in tiles"""
        return self._batch_tile_method(
            "detect_and_extract",
            batch=[t for t in self.tiles if not t.features_detected],
        )

    def align(self, origin=None, limit=None, **kwargs):
        """Builds a mosaic by checking each tile against all others

        Parameters
        ----------
        origin : Tile
            the tile around which to build the mosaic. If not given, method
            will select the tile with the largest number of features.
        limit : int
            the number of tiles that must be successfully placed before the
            method finishes. If not given, the method will continue until
            it runs out of adjacent tiles with matching features. Setting
            a limit allows a decent mosaic to be created quickly.
        kwargs :
            any keyword argument accepted by the align_to method on the
            Tiles comprising this mosaic
        """

        self.detect_and_extract()

        # Limit to tiles with detected features
        candidates = [t for t in self.tiles if t.features_detected]
        if not candidates:
            raise RuntimeError("No features detected in tiles")

        # Start building from feature-rich tiles
        candidates.sort(key=lambda t: -len(t.keypoints))
        tiles = [origin if origin is not None else candidates.pop(0)]

        # Set first tile to 0, 0
        if tiles[0].y is None:
            tiles[0].y = 0
        if tiles[0].x is None:
            tiles[0].x = 0

        while tiles and candidates:

            # Stop aligning if limit is reached
            if limit is not None and self.placed >= limit:
                logger.info(
                    f"Stopped aligning (limit={limit}," f" placed={self.placed})"
                )
                break

            if len(tiles) == 1:
                tile = tiles[0]
            else:
                tile = self.tile_class(tiles[0].draw(tiles[1:]))
                tile.detect_and_extract()

                tile.y = min([t.y for t in tiles])
                tile.x = min([t.x for t in tiles])

            self._batch_tile_method("align_to", tile, batch=candidates, **kwargs)

            # Remove newly placed tiles from candidates
            tiles = [t for t in candidates if t.placed]
            candidates = [t for t in candidates if not t.placed]

        else:
            if limit is not None:
                logger.warning(f"Failed to place {limit} tiles")

        if self.placed == 1 and len(self.tiles) > 1:
            raise RuntimeError("Could not align tiles")

        logger.info(f"Aligned {self.placed} tiles in {self}")

    def build_out(self, *args, **kwargs):
        """Warns user that build_out is not implemented in Mosaic class

        Use StructuredMosaic instead to get this functionality.
        """
        logger.warning("build_out() ignored on Mosaic")

    def smooth_seams(self, origin=None):
        """Smooths intensities at seams between tiles

        Parameters
        ----------
        origin : Tile
            starting tile
        """

        tiles = [self._get_origin() if origin is None else origin]
        placed = [t for t in self.tiles if t.placed and t not in tiles]
        smoothed = []

        min_xtn = 0

        while tiles:

            # Batch align adjacent tiles
            unique = {}
            for tile in tiles:

                # Corner-to-corner matches can give bad results, so
                # exclude small-area matches. This could cause problems
                # with non-gridded tilesets.
                xing = [t for t in placed if t.intersects(tile)]
                if not min_xtn and len(xing) > 4:
                    xtns = [t.intersection(tile)[0].size for t in xing]
                    min_xtn = min(xtns) * 1.5

                xing = [t for t in xing if t.intersection(tile)[0].size > min_xtn]

                for neighbor in xing:
                    if neighbor not in smoothed:
                        unique.setdefault(neighbor.id, (tile, neighbor))

            batch = list(unique.values())
            self._batch_tile_method(
                "match_gamma_to", [t for t, _ in batch], batch=[n for _, n in batch]
            )

            tiles = [n for _, n in batch]
            smoothed.extend(tiles)

        logger.info(f"Smoothed seams in {self}")

    def stitch(self, channel_order=None):
        """Stitches mosaic using either placed tiles or row/col of tiles

        Parameters
        ----------
        channel_order : str
            order of the three color channels in the stitched array, for
            example, RGB. Uses the backend order if not given, which can
            give unexpected results (for example, OpenCV uses BGR).

        Returns
        -------
        numpy.ndarray
        """

        placed = [t for t in self.tiles if t.placed]

        # If mosaic has not been aligned, set x and y based on tile location
        reset_xy = not placed
        if reset_xy:
            y = 0
            for row in self.grid:
                x = 0
                for tile in row:
                    tile.x = x
                    tile.y = y
                    x += tile.width
                y += tile.height
            placed = self.tiles

        self._normalize_coordinates()
        arr = placed[0].draw(placed[1:])

        # Reorder color channels to match the specified order
        if channel_order and channel_order.upper() != placed[0].channel_order:
            order = [placed[0].channel_order.index(c) for c in channel_order.upper()]
            order.extend(range(len(order), len(arr.shape) + 1))
            arr = arr[..., order].copy()

        # Reset tile x and y to None if set above
        if reset_xy:
            for tile in self.tiles:
                tile.y = None
                tile.x = None

        logger.info(f"Drew {self}")

        return arr

    def save(self, path):
        """Saves mosaic to path

        Parameters
        ----------
        path : str
            file path
        """
        self.tile_class.backend_save(path, self.stitch())

    def show(self, *args, **kwargs):
        """Shows the mosaic"""
        self.tile_class.backend_show(self.stitch(), *args, **kwargs)

    def _batch_tile_method(self, method, *args, batch=None, **kwargs):
        """Runs a tile method across many tiles, in f where possible

        Eligible methods must:

        + Be chainable
        + Modify the tile in place
        + Only affect the current tile
        + Use a consistent set of arguments

        Parameters
        ----------
        method : str
            name of a chainable tile method
        *args:
            any argument that can be passed to the method. If passed as a
            list, it should have the same number of items as batch.
        batch : list
            list of tiles. If None, uses self.tiles instead.
        **kwargs:
            any keyword argument that can be passed to the method. If passed
            as a list, it should have the same number of items as batch.W
        """

        if batch is None:
            batch = self.tiles

        if not batch:
            return

        logger.info(
            f"Running tile.{method}() using {self.num_cores}"
            f" cores on batch of {len(batch)} tiles"
        )

        def task(tile, *args, **kwargs):
            return getattr(tile, method)(*args, **kwargs)

        # Each call gets it own set of args/kwargs supplied using using zip,
        # so needs tp make sure the args/kwargs the right size
        targs = [[] for _ in range(len(batch))]
        for arg in args:
            if not isinstance(arg, (list, tuple)) or len(arg) != len(batch):
                arg = [arg] * len(batch)
            for i, arg in enumerate(arg):
                targs[i].append(arg)

        # Ensure that keyword arguments all have same length as batch
        tkwargs = [{} for _ in range(len(batch))]
        for key, val in kwargs.items():
            if not isinstance(val, (list, tuple)) or len(val) != len(batch):
                val = [val] * len(batch)
            for kwdict, val in zip(tkwargs, val):
                kwdict[key] = val

        results = self.pool(
            delayed(task)(t, *a, **k) for t, a, k in zip(batch, targs, tkwargs)
        )

        for tile, result in zip(batch, results):
            tile.update(result)

        logger.info(f"Finished running tile.{method}() ({len(batch)} tiles)")

    def _build_grid(self, path_or_tiles):
        """Builds grid and populates related attributes"""

        tiles = self._get_tiles(path_or_tiles)

        # Create the grid, which is meaningless for unstructured tilesets
        # but provides a convenient way to view them
        if self.grid is None:
            self.shape = self._estimate_shape(tiles)
            self.size = self.shape[0] * self.shape[1]
            self.grid = build_grid(
                tiles, self.shape[1], fill_value=self.placeholder(tiles[0])
            )

        self._verify_tiles()

        # Situate each tile in the grid
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                tile.row = y
                tile.col = x
                tile.grid = self.grid

    def _get_origin(self):
        """Selects a starting tile to work outward from"""

        # If tiles have been placed, choose one that includes the center of
        # the mosaic
        placed = [t for t in self.tiles if t.placed]
        if placed:

            y1, x1, y2, x2 = self.bounds()
            y_cent = (y2 - y1) / 2
            x_cent = (x2 - x1) / 2

            for tile in placed:
                y1, x1, y2, x2 = tile.bounds()
                if y1 <= y_cent <= y2 and x1 <= x_cent <= x2:
                    return tile

        # If no tiles have been placed, choose based on the number of features
        tiles = [t for t in self.tiles if t.features_detected]
        if not tiles:
            raise ValueError(
                "No tiles with detected features found. This can happen"
                " after running detect_and_extract() if reset_tiles()"
                " is used."
            )
        tiles.sort(key=lambda t: -t.size)
        return tiles[0]

    def _get_tiles(self, path_or_tiles):
        """Gets a list of tiles

        Parameters
        ----------
        path_or_tiles : str or list-like
            either the path to a directory containing tiles, a list of Tiles,
            or a list of strings or arrays that can be used to create a Tile

        Returns
        -------
        list of Tile
            list of tiles based on input
        """

        if isinstance(path_or_tiles, str):
            logger.info(
                f"Creating {self.__class__.__name__}" f" from path ({path_or_tiles})"
            )
            ext = self._check_tiles(path_or_tiles)
            items = glob.iglob(os.path.join(path_or_tiles, f"*{ext}"))

        elif is_grid(path_or_tiles):
            logger.info(f"Creating {self.__class__.__name__} from 2D grid")
            items = chain.from_iterable(path_or_tiles)
            self.grid = path_or_tiles.copy()
            self.shape = (len(path_or_tiles), len(path_or_tiles[0]))
            self.size = self.shape[0] * self.shape[1]

        else:
            logger.info(f"Creating {self.__class__.__name__} from 1D list")
            items = path_or_tiles

        # Create and sort list of tiles
        tiles = []
        for item in items:
            if item is None:
                tiles.append(self.placeholder(tiles[0]))
            elif isinstance(item, Tile):
                tiles.append(item.copy())
            else:
                tiles.append(self.tile_class(item))

        # Natural sort tiles if a directory was provided. Tiles provided as
        # lists are kept in their original order.
        if isinstance(path_or_tiles, str):
            tiles.sort()

        return tiles

    def _normalize_coordinates(self):
        """Normalizes coordinates so that origin of the mosaic is (0, 0)"""
        if self.placed:
            min_y, min_x = self.bounds()[:2]
            for tile in self.tiles:
                if tile.placed:
                    tile.y -= min_y
                    tile.x -= min_x

    def _verify_tiles(self):
        """Verifies that tiles and grid are valid and contain same objects"""
        # if None in self.tiles:
        #    raise ValueError("Ragged grids are not allowed")

        tile_ids = {t.id for t in self.tiles}
        for row in self.grid:
            for tile in row:
                if tile.id not in tile_ids:
                    raise ValueError("grid and tiles contain different objects")

    @staticmethod
    def _check_tiles(path):
        """Finds and evaluates image files in the given directory"""
        exts = defaultdict(int)
        for fp in glob.iglob(os.path.join(path, "*.*")):
            if imghdr.what(fp):
                exts[os.path.splitext(fp)[1]] += 1

        if not exts:
            raise ValueError(f"No images found in `{path}`")

        ext = [k for k, _ in sorted(exts.items(), key=lambda kv: -kv[1])][0]
        return ext

    @staticmethod
    def _estimate_shape(tiles):
        """Fits tiles into a shape similar to a square"""
        size = len(tiles)

        widths = []
        for num in range(1, size + 1):
            if not size % num:
                widths.append(num)
        width = widths[len(widths) // 2]
        height = size / width

        shape = [height, width]
        if tiles[0].channels != 1:
            shape.append(tiles[0].channels)

        return tuple([int(n) for n in shape])


class StructuredMosaic(Mosaic):
    """Stitches a mosaic from a list of tiles with a known structure"""

    def __init__(
        self,
        path_or_tiles,
        tile_class=None,
        dim=None,
        origin="upper left",
        direction="horizontal",
        pattern="raster",
    ):
        """Initializes a structured mosaic from a list of tiles

        Parameters
        ----------
        path_or_tiles : str or list-like
            either the path to a directory containing tiles, a list of Tiles,
            or a list of strings or arrays that can be used to create a Tile
        tile_class : class
            class to use for tiles in the mosaic. Defaults to OpenCVTile.
        dim : tuple or int
            either the shape of the mosaic as (height, width) or the number
            of tiles in the direction traversed first, that is, the number of
            columns (if horizontal) or number of rows (if vertical)
        origin : str
            the position of the first tile in the mosaic. One of "upper left",
            "upper right", "lower left", or "lower right".
        direction : str
            direction to traverse first when building the mosaic. Either
            "horizontal" or "vertical".
        pattern : str
            whether the grid is rastered or snaked. Either "raster" or "snake".
        """

        super().__init__(path_or_tiles, tile_class=tile_class)

        tiles = self._get_tiles(path_or_tiles)

        # Create the grid if source is not a grid already
        if self.grid is None:
            self.shape, direction = self._refine_grid_params(tiles, dim, direction)
            self.size = self.shape[0] * self.shape[1]
            self.grid = build_grid(
                tiles,
                self.shape[1] if direction == "horizontal" else self.shape[0],
                origin=origin,
                direction=direction,
                pattern=pattern,
                fill_value=self.placeholder(tiles[0]),
            )

        self._verify_tiles()

        # Situate each tile in the mosaic
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                tile.row = y
                tile.col = x
                tile.grid = self.grid

    def align(self, origin=None, limit=None, **kwargs):
        """Builds a mosaic outward from a single tile using feature matching

        Parameters
        ----------
        origin : Tile
            the tile around which to build the mosaic. If not given, method
            will select a tile near the center of the mosaic.
        limit : int
            the number of tiles that must be successfully placed before the
            method finishes. If not given, the method will continue until
            it runs out of adjacent tiles with matching features. Setting
            a limit allows a decent mosaic to be created quickly.
        kwargs :
            any keyword argument accepted by the align_to method on the
            Tiles comprising this mosaic
        """

        # Align tiles outward from a single tile
        tiles = [self._get_origin() if origin is None else origin]

        # Set origin to 0, 0
        if tiles[0].y is None:
            tiles[0].y = 0
        if tiles[0].x is None:
            tiles[0].x = 0

        while tiles:

            # Stop aligning if limit is reached
            if limit is not None and self.placed >= limit:
                logger.info(f"Stopped aligning (limit={limit}, placed={self.placed})")
                break

            # Batch detect and extract features from tiles
            unique = {}
            for tile in tiles:
                unique.setdefault(tile.id, tile)
                for neighbor in tile.neighbors().values():
                    unique.setdefault(neighbor.id, neighbor)
            batch = [t for t in unique.values() if not t.features_detected]
            self._batch_tile_method("detect_and_extract", batch=batch)

            # Batch align adjacent tiles
            unique = {}
            for tile in tiles:
                for neighbor in tile.neighbors().values():
                    unique.setdefault(neighbor.id, (tile, neighbor))
            batch = [(t, n) for t, n in unique.values() if not n.placed]
            self._batch_tile_method(
                "align_to", [t for t, _ in batch], batch=[n for _, n in batch], **kwargs
            )

            # Otherwise re-run the loop with the next group of tiles
            tiles = [n for _, n in batch if n.placed]

        else:
            if limit is not None:
                logger.warning(f"Failed to place {limit} tiles")

        if self.placed == 1 and len(self.tiles) > 1:
            raise RuntimeError("Could not align tiles")

        logger.info(f"Aligned {self.placed} tiles in {self}")

    def build_out(self, from_placed=True, offsets=None):
        """Builds out from already placed tiles using the given offset

        Used to complete mosaics that include tiles that were not placed
        when the mosaic was built, either because the user assigned a limit
        or because the feature matching algorithm failed to find a home for
        them.

        Parameters
        ----------
        from_placed : bool
            if True, unplaced tiles will be tacked onto already placed tiles
            using the given offsets. If False, a new mosaic will be calculated
            from scratch using the given offsets.
        offsets : tuple
            offsets between adjacent tiles as dy_row, dx_row, dy_col, dx_col.
            If not given, the method will estimate the offsets if any tiles
            have been placed or will ignore offsets if not.
        """

        if offsets is None:
            if [t for t in self.tiles if t.placed]:
                offsets = self._estimate_offset()
            else:
                offsets = (self.tiles[0].height, 0, self.tiles[0].width, 0)

        dy_row, dx_row, dy_col, dx_col = offsets

        # Builds out from already placed tiles using the offsets
        if from_placed:
            tiles = [t for t in self.tiles if t.placed]
            while tiles:
                new = []
                for tile in tiles:
                    for direction, neighbor in tile.neighbors().items():
                        if not neighbor.placed:
                            if direction == "top":
                                neighbor.y = tile.y - dy_row
                                neighbor.x = tile.x - dx_row
                            elif direction == "right":
                                neighbor.y = tile.y + dy_col
                                neighbor.x = tile.x + dx_col
                            elif direction == "bottom":
                                neighbor.y = tile.y + dy_row
                                neighbor.x = tile.x + dx_row
                            elif direction == "left":
                                neighbor.y = tile.y - dy_col
                                neighbor.x = tile.x - dx_col
                            new.append(neighbor)
                    tiles = new

            logger.info(
                f"Built {self} out from previously placed" f" tiles (offsets={offsets})"
            )

        # Builds mosaic from scratch based on the offsets
        else:
            for tile in self.tiles:
                tile.y = tile.row * dy_row + tile.col * dy_col
                tile.x = tile.row * dx_row + tile.col * dx_col

            logger.info(f"Rebuilt {self} (offsets={offsets})")

    def _build_grid(self, path_or_tiles):
        """Builds grid and populates related attributes"""

    def _estimate_offset(self):
        """Estimates offset based on row and column

        Returns
        -------
        tuple
            average offsets between adjacent tiles as dy_row (change in y
            between rows), dx_row (change in x between rows), dy_col (change
            in y between columns), dx_col (change in x between columns)
        """

        placed = [t for t in self.tiles if t.placed]

        # Test size occupied by the placed tiles. If the tiles don't
        # span at least two rows/columns, the offset calculation will
        # give a bad result.
        ys = []
        xs = []
        for tile in placed:
            y1, x1, y2, x2 = tile.bounds(as_int=True)
            ys.extend([y1, y2])
            xs.extend([x1, x2])

        height = max(ys) - min(ys)
        width = max(xs) - min(xs)

        tile = self.tiles[0]
        if height < (tile.height * 1.5) or width < (tile.width * 1.5):
            logger.warning(
                "Placed tiles may be too close together to calculate an"
                " accurate offset. If you specified a limit when running"
                " align(), you may want to try a larger value."
            )

        # Get minimum row/col indexes so can normalize these to zero below
        min_row = min([t.row for t in placed])
        min_col = min([t.col for t in placed])

        # Normalize coordinates to positive values starting at 0, 0
        self._normalize_coordinates()

        # Calculate average offsets based on row and column by looking at
        # where tiles have been placed
        a = []
        by = []
        bx = []
        for tile in placed:
            a.append([tile.row - min_row, tile.col - min_col])
            by.append(tile.y)
            bx.append(tile.x)

        a = np.array(a)
        by = np.array(by)
        bx = np.array(bx)

        dy_row, dy_col = np.linalg.lstsq(a, by, rcond=None)[0]
        dx_row, dx_col = np.linalg.lstsq(a, bx, rcond=None)[0]

        logger.info(
            f"Estimated offsets in {self} as" f" {(dy_row, dx_row, dy_col, dx_col)}"
        )

        return dy_row, dx_row, dy_col, dx_col

    def _get_origin(self):
        """Selects a starting tile to work outward from"""
        return self.grid[self.shape[0] // 2][self.shape[1] // 2]

    @staticmethod
    def _refine_grid_params(tiles, dim, direction):
        """Refines shape and direction supplied during initialization"""

        if dim is None or isinstance(dim, int):
            if direction == "horizontal":
                dim = (None, dim)
            else:
                dim = (dim, None)

        height, width = dim
        size = len(tiles)

        if not width and not height:

            # Check for file naming pattern used by the Mineral Sciences SEM
            cols = []
            for tile in tiles:
                try:
                    cols.append(int(re.search(r"@(\d+)", tile.source).group(1)))
                except (AttributeError, TypeError):
                    pass

            if not cols:
                raise ValueError("Could not infer shape of StructuredMosaic")

            width = len(range(min(cols), max(cols) + 1))
            direction = "vertical"

        if width and not height:
            while size % width:
                size += 1
            height = size // width

        elif height and not width:
            while size % height:
                size += 1
            width = size // height

        shape = [height, width]
        if tiles[0].channel_axis:
            shape.append(tiles[0].channels)

        return tuple([int(n) for n in shape]), direction


def build_grid(
    items,
    dim,
    origin="upper left",
    direction="horizontal",
    pattern="raster",
    fill_value=None,
):
    """Builds a grid from a list

    Parameters
    ----------
    items : list
        list to convert to a grid
    dim : tuple or int
        either the shape of the mosaic as (height, width) or the number
        of tiles in the direction traversed first, that is, the number of
        columns (if horizontal) or number of rows (if vertical)
    origin : str
        the position of the first tile in the mosaic. One of "upper left",
        "upper right", "lower left", or "lower right".
    direction : str
        direction to traverse first when building the mosaic. Either
        "horizontal" or "vertical".
    pattern : str
        whether the grid is a raster or snake
    fill_value :
        value used to fill missing items in a ragged grid

    Returns
    -------
    list
        List of rows in the grid
    """

    # Validate keywords
    origins = {
        "ul": "upper left",
        "ur": "upper right",
        "lr": "lower right",
        "ll": "lower left",
    }
    origin = origins.get(origin, origin)
    if origin not in origins.values():
        raise ValueError(f"origin must be a key or value in {origins}")

    directions = ("horizontal", "vertical")
    if direction not in directions:
        raise ValueError(f"direction must be one of {directions}")

    patterns = ("raster", "snake")
    if pattern not in patterns:
        raise ValueError(f"pattern must be one of {patterns}")

    # Work from a copy of the list so that original remains intact
    items = items.copy()

    # Get expected dimensions of the mosaic
    try:
        dim_primary, dim_secondary = dim
    except TypeError:
        dim_primary = dim
        dim_secondary = None

    # Pad list so that all rows/columns have a full allotment of tiles
    while len(items) % dim_primary:
        items.append(None)

    # Replace Nones with the given fill value
    if None in items:
        logger.warning("Grid is ragged")
        if fill_value is not None:
            for i, item in enumerate(items):
                if item is None:
                    try:
                        items[i] = fill_value.copy()
                    except AttributeError:
                        items[i] = fill_value

    # Split items into chunks of the given size. Use indexes instead of
    # directly using the array so that arrays of arrays can be gridded
    # without getting a warning from numpy.
    rows = []
    for row in np.array_split(np.arange(len(items)), len(items) / dim_primary):
        rows.append([items[i] for i in row])

    # Test if mosaic is expected size
    if dim_secondary and len(rows) != dim_secondary:
        raise ValueError("Grid is wrong shape")

    # Reverse every other row if snake pattern
    if pattern == "snake":
        for i, row in enumerate(rows):
            if i and i % 2:
                row.reverse()

    # Extract rows from columns if using vertical instead of horizontal
    if direction == "vertical":
        cols = rows
        rows = []
        for i in range(len(cols[0])):
            row = []
            for col in cols:
                row.append(col[i])
            rows.append(row)

    # Reverse row order if origin is lower left or right corner
    if "lower" in origin:
        rows.reverse()

    # Reverse order of individual rows if origin is on the right
    if "right" in origin:
        for row in rows:
            row.reverse()

    return rows


def create_mosaic(
    path_or_tiles, tile_class=None, dim=None, origin=None, direction=None, pattern=None
):
    """Creates a mosaic

    See StructuredMosaic for available parameters.

    Returns
    -------
    Mosaic or StructuredMosaic
        tiles as either a structured or unstructured mosaic
    """

    kwargs = {"dim": dim, "origin": origin, "direction": direction, "pattern": pattern}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        return StructuredMosaic(path_or_tiles, tile_class=tile_class, **kwargs)
    except ValueError:
        if kwargs:
            raise
        return Mosaic(path_or_tiles, tile_class=tile_class)


def is_grid(items):
    """Tests if an iterable looks like a grid

    Parameters
    ----------
    items : list-like
        list of items

    Returns
    -------
    bool
        True if tiles look like a grid, False if not
    """

    # Both items and all direct children of items must be iterable
    try:
        iter(items)
        iter(items[0])
    except (IndexError, TypeError):
        return False

    # Strings are iterable, so rule them out too
    if isinstance(items[0], str):
        return False

    # Check if all rows have the same length
    if len({len(r) for r in items}) > 1:
        return False

    # Test if individual items are or can be made into tiles
    val = items[0][0]
    if isinstance(val, (str, Tile)) or (
        isinstance(val, np.ndarray) and 2 <= len(val.shape) <= 3
    ):
        return True

    return False
