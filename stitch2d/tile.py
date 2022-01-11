"""Reads and helps place a single image from a 2D grid"""
import logging
import os
import re
from tempfile import NamedTemporaryFile
import uuid

import cv2 as cv
import numpy as np
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.exposure import adjust_gamma
from skimage.feature import SIFT, match_descriptors
from skimage.transform import resize


logger = logging.getLogger(__name__)


class _DefaultInstance:
    """Creates new instance of a class using user-provided defaults

    Allows non-pickleable objects (like those in OpenCV) to be pickled by
    storing classes and parameters instead of the objects themselves.
    """

    def __init__(self, obj, *args, cache=False, **kwargs):
        self._class = obj
        self._args = args
        self._kwargs = kwargs

        self._cached = obj(*args, **kwargs) if cache else None

    def __call__(self):
        """Creates a new instance using the supplied parameters"""
        if self._cached:
            return self._cached
        return self._class(*self._args, **self._kwargs)


class Tile:
    """An image tile in a mosaic

    Attributes
    ----------
    source : str or array-like
        the original data used to created the tile. Either the path to an
        image file or an array with 1, 3, or 4 channels.
    imdata : numpy.ndarray
        image data
    id : str
        a UUID uniquely identifying the tile
    row : int
        the index of the row where the tile appears in the mosaic
    col : int
        the index of the column where the tile appears in the mosaic
    y : float
        the y coordinate of the image within the mosaic
    x : float
        the x coordinate of the image within the mosaic
    scale : float
        the current scale of the tile relative to the original image
    features_detected : bool
        whether any features were detected in this image
    descriptors : numpy.ndarray
        list of descriptors found in this image
    keypoints : numpy.ndarray
        list of coordinates of descriptors found in this image
    """

    #: dict : maps strings to a subclass-specific feature detector
    detectors = {}

    #: dict : maps strings to a subclass-specific feature matcher
    matchers = {}

    def __init__(self, data, detector="sift"):
        """Initializes a mosaic from a list of tiles

        Parameters
        ----------
        data : str or numpy.ndarray
            path to an image file or an array of image data
        detector : str
            name of the detector used to find/extract features. Currently
            only sift is supported.
        """

        self.id = uuid.uuid4()

        self.source = data if isinstance(data, str) else data.copy()
        self.imdata = self.load_imdata()

        if self.imdata is None:
            raise IOError(f"No image data found (source={self.source})")

        self.row = None
        self.col = None

        self.y = None
        self.x = None

        self.scale = 1.0

        self.grid = None
        self.is_placeholder = False

        self.features_detected = None
        self.descriptors = None
        self.keypoints = None

        self._detector = detector

    def __str__(self):
        return f"<{self.__class__.__name__} id={self.id} shape={self.shape}>"

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.shape == other.shape
            and self.size == other.size
            and self.dtype == other.dtype
            and self.placed == other.placed
            and self.row == other.row
            and self.col == other.col
            and self.y == other.y
            and self.x == other.x
            and self.scale == other.scale
            and self.features_detected == other.features_detected
            and np.array_equal(self.imdata, other.imdata)
            and (
                self.descriptors is None
                and other.descriptors is None
                or np.array_equal(self.descriptors, other.descriptors)
            )
            and (
                self.keypoints is None
                and other.keypoints is None
                or np.array_equal(self.keypoints, other.keypoints)
            )
        )

    def __lt__(self, other):
        """Natural sorts tiles by file name if loaded from a file"""
        if isinstance(self.source, str) and isinstance(other.source, str):
            tiles = []
            for tile in [self, other]:
                sortable = []
                for part in re.split(r"(\d+)", tile.source):
                    sortable.append(int(part) if part.isnumeric() else part.lower())
                tiles.append(sortable)
            return tiles[0] < tiles[1]

        # Can only sort Tiles created from a file
        return False

    @property
    def detector(self):
        """Gets the detector used to align this tile to another tile"""
        if isinstance(self._detector, str):
            try:
                return self.detectors[self._detector]()
            except KeyError as exc:
                raise KeyError(
                    f"detector must be one of {self.detectors.keys()}"
                ) from exc
        return self._detector

    @detector.setter
    def detector(self, val):
        self._detector = val

    @property
    def matcher(self):
        """Gets the matcher used to align this tile to another tile"""
        if isinstance(self._matcher, str):
            try:
                return self.matchers[self._matcher]()
            except KeyError as exc:
                raise KeyError(
                    f"matcher must be one of {self.matchers.keys()}"
                ) from exc
        return self._matcher

    @matcher.setter
    def matcher(self, val):
        self._matcher = val

    @property
    def height(self):
        """Gets the height of the image in pixels"""
        return self.shape[0]

    @property
    def width(self):
        """Gets the width of the image in pixels"""
        return self.shape[1]

    @property
    def channels(self):
        """Gets the number of channels in the image"""
        try:
            return self.shape[2]
        except IndexError:
            return 1

    @property
    def channel_axis(self):
        """Gets the index where channel info is stored"""
        return 2 if 3 <= len(self.shape) <= 4 else None

    @property
    def dtype(self):
        """Gets the dtype of the image"""
        return self.imdata.dtype

    @property
    def shape(self):
        """Gets the shape of the image"""
        return self.imdata.shape

    @property
    def size(self):
        """Gets the size of the image"""
        return self.imdata.size

    @property
    def mp(self):
        """Gets the size of the image in megapixels"""
        return self.imdata.size / 1e6

    @property
    def placed(self):
        """Whether the tile has been assigned coordinates in the mosaic"""
        return self.y is not None and self.x is not None

    def load_imdata(self):
        """Loads copy of source data

        Returns
        -------
        numpy.ndarray
            copy of source data
        """
        raise NotImplementedError("`load_imdata` must be implemented in subclass")

    def copy(self):
        """Creates a copy of the tile

        Parameters
        ----------
        grid: list of lists
            grid from the mosaic containing the tile

        Returns
        -------
        Mosaic
            copy of the tile
        """
        source = self.source
        if not isinstance(self.source, str):
            source = self.source.copy()

        copy = self.__class__(source)

        copy.imdata = self.imdata.copy()
        copy.scale = self.scale

        copy.row = self.row
        copy.col = self.col

        copy.y = self.y
        copy.x = self.x

        copy.grid = None

        copy.features_detected = self.features_detected

        if copy.features_detected:
            copy.descriptors = self.descriptors
            copy.keypoints = self.keypoints

        return copy

    def bounds(self, as_int=False):
        """Calculates the position of the tile within the mosaic

        Parameters
        ----------
        as_int : bool
            whether bounds are converted to integers before returning

        Returns
        -------
        tuple
            bounds of the image in the mosaic coordinate system as
            (y1, x1, y2, x2)
        """
        bounds = self.y, self.x, self.y + self.height, self.x + self.width
        if as_int:
            bounds = [int(n) for n in bounds]
        return tuple(bounds)

    def neighbors(self):
        """Finds adjacent tiles

        Parameters
        ----------
        y : int
            row index
        x : int
            column index

        Returns
        -------
        dict
            neighboring tiles keyed to direction (top, right, bottom, left)
        """
        neighbors = {}
        for direction, (y, x) in {
            "top": (self.row - 1, self.col),
            "right": (self.row, self.col + 1),
            "bottom": (self.row + 1, self.col),
            "left": (self.row, self.col - 1),
        }.items():
            if y >= 0 and x >= 0:
                try:
                    neighbors[direction] = self.grid[y][x]
                except IndexError:
                    pass

        return neighbors

    def convert_mosaic_coords(self, y1, x1, y2, x2):
        """Converts mosaic coordinates to image coordinates

        Returns
        -------
        tuple
            mosaic coordinates translated to image coordinates
        """
        return (int(y1 - self.y), int(x1 - self.x), int(y2 - self.y), int(x2 - self.x))

    def update(self, other):
        """Updates attributes to match another tile

        Parameters
        ----------
        other : Tile
            a tile with attributes to copy over to this one
        """
        for attr in (
            "imdata",
            "source",
            "row",
            "col",
            "y",
            "x",
            "scale",
            "features_detected",
            "descriptors",
            "keypoints",
        ):
            setattr(self, attr, getattr(other, attr))

    def crop(self, box, convert_mosaic_coords=True):
        """Crops tile to the given box

        Parameters
        ----------
        box : tuple
            box to crop to as (y1, x1, y2, x2)
        convert_mosaic_coords : bool
            whether to convert the given coordinates from mosaic to image
            coordinates

        Returns
        -------
        numpy.ndarray
            image data cropped to the given box
        """
        if convert_mosaic_coords:
            box = self.convert_mosaic_coords(*box)
        y1, x1, y2, x2 = box
        return self.imdata.copy()[y1:y2, x1:x2]

    def intersection(self, other):
        """Finds the intersection between two placed tiles

        Parameters
        ----------
        other : Tile
            an adjacent tile that has already been placed in the mosaic

        Returns
        -------
        tuple of Tile
            the overlapping portion of both tiles
        """

        # Based on https://stackoverflow.com/a/25068722
        sy1, sx1, sy2, sx2 = self.bounds()
        oy1, ox1, oy2, ox2 = other.bounds()

        y1 = max(min(sy1, sy2), min(oy1, oy2))
        x1 = max(min(sx1, sx2), min(ox1, ox2))
        y2 = min(max(sy1, sy2), max(oy1, oy2))
        x2 = min(max(sx1, sx2), max(ox1, ox2))

        if x1 >= x2 or y1 >= y2:
            raise ValueError("Tiles do not intersect")

        xtn1 = self.crop((y1, x1, y2, x2), convert_mosaic_coords=True)
        xtn2 = other.crop((y1, x1, y2, x2), convert_mosaic_coords=True)

        return xtn1, xtn2

    def intersects(self, other):
        """Tests if two placed tiles intersect

        Parameters
        ----------
        other : Tile
            an adjacent tile that has already been placed in the mosaic

        Returns
        -------
        bool
            True if tiles intersect, False otherwise
        """
        try:
            self.intersection(other)
            return True
        except ValueError:
            return False

    def reset(self):
        """Restores original image and resets coordinate and feature attrs

        Returns
        -------
        Tile
            the original tile updated to restore the original image data
        """

        self.imdata = self.load_imdata()

        if self.x:
            self.x /= self.scale
        if self.y:
            self.y /= self.scale

        self.scale = 1.0

        self.features_detected = None
        self.descriptors = None
        self.keypoints = None

        return self

    def match_gamma_to(self, other):
        """Scales intensity to match intersecting region of another tile

        Parameters
        ----------
        other : Tile
            a tile that intersects this one

        Returns
        -------
        Tile
            the original tile with its intensity modified
        """

        def find_gamma(im, other):
            """Finds gamma that best matches image to a reference image"""

            increment = 0.02
            if np.mean(im) < np.mean(other):
                increment *= -1

            gamma = 1
            gammas = {}
            while 0 < gamma < 5:

                im_adj = adjust_gamma(im, gamma)

                # Accept when ratio between tiles is within 2%
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.mean(im_adj) / np.mean(other)

                if 0.98 < ratio < 1.02:
                    return gamma

                # Capture failed values in case no results fall within 2%
                gammas[abs(1 - ratio)] = gamma

                gamma += increment

            return gammas[min(gammas)]

        xtn_self, xtn_other = self.intersection(other)
        gamma = find_gamma(xtn_self, xtn_other)
        self.imdata = adjust_gamma(self.imdata, gamma)

        return self

    def draw(self, others=None):
        """Creates an image from the provided tiles

        Parameters
        ----------
        others : list of Tiles
            a list of tiles to include in the new image. Only tiles that
            have been placed will be included.

        Returns
        -------
        numpy.ndarray
            an image including all provided tiles
        """

        if others is None:
            return self.imdata

        tiles = [t for t in [self] + others if t.placed]

        ys = []
        xs = []
        for tile in tiles:
            y1, x1, y2, x2 = tile.bounds(as_int=True)
            ys.extend([y1, y2])
            xs.extend([x1, x2])

        height = max(ys) - min(ys)
        width = max(xs) - min(xs)

        shape = [height, width]
        if self.channel_axis:
            shape.append(self.channels)

        arr = np.zeros(shape, dtype=self.dtype)

        # Sort placeholders to front of list so they're drawn first. Tiles
        # with image data will then overwrite them where they overlap.
        tiles = [t for t in [self] + others if t.is_placeholder]
        tiles.extend([t for t in [self] + others if not t.is_placeholder])

        for tile in tiles:
            y = int(tile.y - min(ys))
            x = int(tile.x - min(xs))

            arr[y : y + tile.height, x : x + tile.width] = tile.imdata

        return arr

    def save(self, path, others=None):
        """Saves an image created from the provided tiles

        Parameters
        ----------
        path : str
            file path
        others : list of Tiles
            a list of tiles to include in the new image. Only tiles that
            have been placed will be included.
        """
        return self.backend_save(path, self.draw(others))

    def show(self, others=None):
        """Shows an image created from the provided tiles

        Parameters
        ----------
        others : list of Tiles
            a list of tiles to include in the new image. Only tiles that
            have been placed will be included.
        """
        return self.backend_show(self.draw(others))

    def gray(self):
        """Returns copy of image converted to grayscale

        Returns
        -------
        numpy.ndarray
            grayscale version of the original iamge
        """
        raise NotImplementedError("`gray` must be implemented in subclass")

    def resize(self, size_or_shape, *args, **kwargs):
        """Resizes image to a given size or shape

        Parameters
        ----------
        size_or_shape : float, int, or tuple of ints
            size in megapixels or shape of resized image
        *args :
            any argument accepted by the resize function used by the subclass
        **kwargs :
            any keyword argument accepted by the resize function used by the
            subclass
        """
        raise NotImplementedError("`resize` must be implemented in subclass")

    def downsample(self, size_or_shape, *args, **kwargs):
        """Downsamples image to a given size or shape if smaller than original

        Parameters
        ----------
        size_or_shape : float, int, or tuple of ints
            size in megapixels or shape of resized image as (height, width)
        *args :
            any argument accepted by the resize method of the subclass
        **kwargs :
            any keyword argument accepted by the resize method of the subclass

        Returns
        -------
        Tile
            the original tile downsampled to the given size or shape
        """
        (height, width), _ = self._calc_resized(size_or_shape, *args, **kwargs)
        if height * width < self.size:
            self.resize(size_or_shape, *args, **kwargs)
        return self

    def detect_and_extract(self, *args, **kwargs):
        """Detects and extracts features within the tile

        Parameters
        ----------
        *args :
            any argument accepted by the feature detection method on the
            detector
        **kwargs :
            any keyword argument accepted by the feature detection method
            on the detector
        """
        raise NotImplementedError(
            "`detect_and_extract` must be implemented in subclass"
        )

    def align_to(self, other, **kwargs):
        """Aligns tile to another, already placed tile

        Parameters
        ----------
        other : Tile
            a tile that has already been placed in the mosaic
        """
        raise NotImplementedError("`align_to` must be implemented in subclass")

    @staticmethod
    def backend_save(path, im):
        """Saves image to path using the tile backend

        Parameters
        ----------
        path : str
            file path
        im : numpy.ndarray
            image data
        """
        raise NotImplementedError("`backend_save` must be implemented in subclass")

    @staticmethod
    def backend_show(im):
        """Shows an image using the tile backend

        Parameters
        ----------
        im : numpy.ndarray
            image data
        """
        raise NotImplementedError("`backend_show` must be implemented in subclass")

    def _calc_resized(self, size_or_shape):
        """Calculates shape and scale of a resized image

        Parameters
        ----------
        size_or_shape : float, int, or tuple of ints
            size in megapixels or shape of resized image

        Returns
        -------
        tuple
            shape as (height, width) and scale of resized image
        """

        if isinstance(size_or_shape, (float, int)):
            scale = (size_or_shape * 1e6 / (self.height * self.width)) ** 0.5
            height = int(self.height * scale)
            width = int(self.width * scale)
        else:
            height, width = size_or_shape
            scale = ((height * width) / (self.height * self.width)) ** 0.5

        return (height, width), scale

    @staticmethod
    def _within_n_pixels(y, x, n_pixels=5):
        """Calculates fraction of coordinates within given range of median

        Parameters
        ----------
        y : list
            list of y coordinates
        x : list
            list of x coordinates. Same length as y.
        n_pixels : int
            maximum number of pixels coordinate can be from the median

        Returns
        -------
        float
            fraction of coordinates within range in both y and x
        """

        min_y = np.median(y) - n_pixels
        max_y = np.median(y) + n_pixels

        min_x = np.median(x) - n_pixels
        max_x = np.median(x) + n_pixels

        within = [
            (y, x) for y, x in zip(y, x) if min_y <= y <= max_y and min_x <= x <= max_x
        ]

        return len(within) / len(y)


class OpenCVTile(Tile):
    """An image tile in a mosaic loaded and manipulated using OpenCV

    See Tile for available attributes.
    """

    detectors = {"sift": _DefaultInstance(cv.SIFT_create)}

    matchers = {
        "bf": _DefaultInstance(cv.BFMatcher),
        "flann": _DefaultInstance(
            cv.FlannBasedMatcher, {"algorithm": 1, "trees": 5}, {"checks": 50}
        ),
    }

    def __init__(self, data, detector="sift", matcher="flann"):
        super().__init__(data)

        self._detector = detector
        self._matcher = matcher

    def load_imdata(self):
        """Loads copy of source data

        Returns
        -------
        numpy.ndarray
            copy of source data
        """
        if isinstance(self.source, str):
            imdata = cv.imread(self.source, cv.IMREAD_UNCHANGED)

            # OpenCV can't read files with non-ASCII characters in Windows.
            # Create a temporary file with the same data as a workaround.
            if imdata is None:

                with NamedTemporaryFile(delete=False) as tmp:
                    with open(self.source, "rb") as f:
                        tmp.write(f.read())

                imdata = cv.imread(tmp.name, cv.IMREAD_UNCHANGED)
                os.unlink(tmp.name)

            return imdata

        return self.source.copy()

    def gray(self):
        """Returns copy of image converted to grayscale

        Returns
        -------
        numpy.ndarray
            grayscale version of the original iamge
        """
        return cv.cvtColor(self.imdata, cv.COLOR_BGR2GRAY)

    def resize(self, size_or_shape, *args, **kwargs):
        """Resizes image to a given size or shape

        Parameters
        ----------
        size_or_shape : float, int, or tuple of ints
            size in megapixels or shape of resized image as (height, width)
        *args :
            any argument accepted by cv.resize
        **kwargs :
            any keyword argument accepted by cv.resize

        Returns
        -------
        OpenCVTile
            the original tile resized to the given size or shape
        """
        if self.scale != 1.0:
            self.reset()

        (height, width), scale = self._calc_resized(size_or_shape)

        if (height, width) != self.shape:
            self.imdata = cv.resize(self.imdata, (width, height), *args, **kwargs)
            self.scale = scale
            if self.placed:
                self.x *= self.scale
                self.y *= self.scale

        return self

    def detect_and_extract(self, *args, **kwargs):
        """Detects and extracts features within the tile

        Parameters
        ----------
        *args :
            any argument accepted by the detect_and_extract method on the
            detector
        **kwargs :
            any keyword argument accepted by the detect_and_extract method
            on the detector

        Returns
        -------
        OpenCVTile
            the original tile updated with features and keypoints
        """

        if self.features_detected is None:

            try:
                detected = self.detector.detectAndCompute(self.imdata, None)
            except KeyError:
                self.features_detected = False
            else:
                self.keypoints, self.descriptors = detected
                self.features_detected = self.descriptors is not None

        return self

    def align_to(self, other, **kwargs):
        """Aligns tile to another, already placed tile

        Parameters
        ----------
        other : Tile
            a tile that has already been placed in the mosaic

        Returns
        -------
        OpenCVTile
            the original tile updated with x and y coordinates
        """

        if self.features_detected and other.features_detected:

            kwargs.setdefault("k", 2)
            matches = self.matcher.knnMatch(
                self.descriptors, other.descriptors, **kwargs
            )

            # Ratios from OpenCV Stitcher docs
            matches = [m for m, n in matches if m.distance < 0.65 * n.distance]

            if len(matches) >= 10:
                dy = []
                dx = []
                for match in matches:

                    x1, y1 = self.keypoints[match.queryIdx].pt
                    x2, y2 = other.keypoints[match.trainIdx].pt

                    dy.append(y1 - y2)
                    dx.append(x1 - x2)

                if self._within_n_pixels(dy, dx, 5) > 0.5:
                    self.y = other.y - np.median(dy)
                    self.x = other.x - np.median(dx)

        return self

    @staticmethod
    def backend_save(path, im):
        """Saves image to path using OpenCV

        Parameters
        ----------
        path : str
            file path
        im : numpy.ndarray
            image data
        """
        cv.imwrite(path, im)

    @staticmethod
    def backend_show(im, title="OpenCV Image"):
        """Shows an image using OpenCV

        Parameters
        ----------
        im : numpy.ndarray
            image data
        """
        cv.imshow(title, im)
        while True:
            cv.waitKey(100)
            if cv.getWindowProperty(title, cv.WND_PROP_VISIBLE) < 1:
                break
        cv.destroyAllWindows()


class ScikitImageTile(Tile):
    """An image tile in a mosaic loaded and manipulated using scikit-image

    See Tile for available attributes.
    """

    def __init__(self, data, detector="sift"):
        self.detectors = {
            "sift": _DefaultInstance(SIFT, cache=True),
        }
        super().__init__(data, detector=detector)

    def load_imdata(self):
        """Loads copy of source data

        Returns
        -------
        numpy.ndarray
            copy of source data
        """
        if isinstance(self.source, str):
            return io.imread(self.source)
        return self.source.copy()

    def gray(self):
        """Returns copy of image converted to grayscale

        Returns
        -------
        numpy.ndarray
            grayscale version of the original iamge
        """
        imdata = self.imdata.copy()
        if self.channels == 4:
            imdata = rgba2rgb(imdata)
        if self.channels == 3:
            imdata = rgb2gray(imdata)
        return imdata

    def resize(self, size_or_shape, *args, **kwargs):
        """Resizes image to a given size or shape

        Parameters
        ----------
        size_or_shape : float, int, or tuple of ints
            size in megapixels or shape of resized image
        *args :
            any argument accepted by skimage.transform.resize
        **kwargs :
            any keyword argument accepted by skimage.transform.resize

        Returns
        -------
        ScikitImageTile
            the original tile resized to the given size of shape
        """
        if self.scale != 1.0:
            self.reset()

        (height, width), scale = self._calc_resized(size_or_shape)

        if (height, width) != self.shape:
            self.imdata = resize(self.imdata, (height, width), *args, **kwargs)
            self.scale = scale
            if self.placed:
                self.x *= self.scale
                self.y *= self.scale

        return self

    def detect_and_extract(self, *args, **kwargs):
        """Detects and extracts features within the tile

        Parameters
        ----------
        *args :
            any argument accepted by the detect_and_extract method on the
            detector
        **kwargs :
            any keyword argument accepted by the detect_and_extract method
            on the detector

        Returns
        -------
        ScikitImageTile
            the original tile updated with features and keypoints
        """

        if self.features_detected is None:
            try:
                self.detector.detect_and_extract(self.gray())
                self.descriptors = self.detector.descriptors
                self.keypoints = self.detector.keypoints
                self.features_detected = self.descriptors.any() and self.keypoints.any()
            except RuntimeError:
                self.features_detected = False

        return self

    def align_to(self, other, **kwargs):
        """Aligns tile to another, already placed tile

        Parameters
        ----------
        other : Tile
            a tile that has already been placed in the mosaic

        Returns
        -------
        ScikitImageTile
            the original tile updated with x and y coordinates
        """
        if self.features_detected and other.features_detected:

            # Ratios from OpenCV Stitcher docs. Other defaults based on
            # match_descriptors example from skimage docs.
            kwargs.setdefault("max_ratio", 0.65)
            kwargs.setdefault("cross_check", True)

            matches = match_descriptors(self.descriptors, other.descriptors, **kwargs)

            if len(matches) >= 10:
                dy = []
                dx = []
                for i, j in matches:
                    y1, x1 = self.keypoints[i]
                    y2, x2 = other.keypoints[j]

                    dy.append(y1 - y2)
                    dx.append(x1 - x2)

                if self._within_n_pixels(dy, dx, 5) > 0.5:
                    self.y = other.y - np.median(dy)
                    self.x = other.x - np.median(dx)

        return self

    @staticmethod
    def backend_save(path, im):
        """Saves image to path using skimage

        Parameters
        ----------
        path : str
            file path
        im : numpy.ndarray
            image data
        """
        io.imsave(path, im)

    @staticmethod
    def backend_show(im):
        """Shows an image using skimage

        Parameters
        ----------
        im : numpy.ndarray
            image data
        """
        io.imshow(im)
        io.show()
