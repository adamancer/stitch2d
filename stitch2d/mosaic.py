"""A Python script used to stitch a two-dimensional tileset into a mosaic.
It includes functions to test and sort the tilset and to determine the
placement of tiles within the final mosaic. Install with
:code:`pip install stitch2d`.

The easiest way to stitch a tileset is to use the
:py:func:`~Stitch2D.Mosaic.mosey` function, which is accessible
from the command line: :code:`stitch2d mosaic`. Use the -h flag to
see additional options.
"""

import csv
import glob
import os
import shlex
import shutil
import subprocess
import sys
import time
import Tkinter
import tkFileDialog
import cPickle as pickle
from copy import copy
from datetime import datetime
from textwrap import fill

try:
    import cv2
except ImportError:
    pass
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .helpers import cluster, cprint, mandolin, mogrify, prompt
from .offset import OffsetEngine




IMAGE_MAP = {
    '.jpg' : 'JPEG',
    '.tif' : 'TIFF',
    '.tiff' : 'TIFF'
}

IMAGE_TYPES = {
    'unspecified' : 'unspecified image type',
    'ref' : 'petrographic microscope, reflected light',
    'rfl' : 'petrographic microscope, reflected light',
    'ppl' : 'petrographic microscope, transmitted light',
    'trans' : 'petrographic microscope, transmitted light',
    'xpl' : 'petrographic microscope, cross-polarized light',
    'xpol' : 'petrographic microscope, cross-polarized light',
    'bsed' : 'SEM, backscatter',
    'nbsed' : 'SEM, normalized backscatter',
    'etd' : 'SEM, secondary electron',
    'sed' : 'SEM, secondary electron',
    'cl' : 'cathodoluminescence',
    'Al' : 'SEM x-ray map, Al',
    'Ca' : 'SEM x-ray map, Ca',
    'Cr' : 'SEM x-ray map, Cr',
    'Fe' : 'SEM x-ray map, Fe',
    'K' : 'SEM x-ray map, K',
    'Mg' : 'SEM x-ray map, Mg',
    'Mn' : 'SEM x-ray map, Mn',
    'Na' : 'SEM x-ray map, Na',
    'S' : 'SEM x-ray map, S',
    'Si' : 'SEM x-ray map, Si',
    'Ti' : 'SEM x-ray map, Ti',
}




class Counter(dict):

    def add(self, key, val=1):
        try:
            self[key] += val
        except KeyError:
            self[key] = val




class Mosaic(object):
    """Contains functions and metadata needed to create a mosaic from a tilset

    Class attributes describe the tiles and tileset and are
    populated using :py:func:`~Stitch2D.mosaic.populate_tiles`.
    To make a mosaic, use :py:func:`~Stitch2D.mosaic.prepare_mosaic`
    to calculate the coordinates, then pass the coordinates to
    :py:func:`~Stitch2D.mosaic.create_mosaic` to stitch.

    Attributes:
        grid (list): specifies position of tiles in grid
        dim (tuple): number of (columns, rows) in the tileset
        size (tuple): (width, height) of individual tiles
        mag (float): magnification of images
        snake (bool): specifies snake pattern
        coordinates (dict): coordinates of tiles in mosaic keyed
            to filepaths
        keypoints (dict): keypoints detected by OpenCV keyed to
            filepaths
    """

    def __init__(self, path, param_file=None, skip_file=None):
        """Initialize new Tileset

        The heavy lifting is done by
        :py:funct:`~Stitch2d.Mosaic.populate_tiles()`,
        which populates and processes the tileset.

        Args:
            path (str): path to tileset
            param_file (str): filepath to parameters file
            skip_file (str): path to file containing indices of skipped tiles
        """
        self.basepath = os.path.dirname(__file__)
        self.normal = True
        self.verbose = True

        self.populate_tiles(path, '.tif', param_file, skip_file)




    def populate_tiles(self, path, ext, param_file=None, skip_file=None):
        """Test, characterize, sort and patch tiles from path

        Args:
            path (str): filepath to tiles
            ext (str): extension on image files
            param_file (str): filepath to parameters file
            skip_file (str): filepath to text file containing the
                list of skipped indices

        Returns:
            None
        """
        # Get descriptive name of tileset based on filename
        self.filename = unicode(os.path.basename(path))
        try:
            self.name, image_type = self.filename.rsplit('_', 1)
            self.name += ' ({})'.format(IMAGE_TYPES[image_type.lower()])
        except (KeyError, ValueError):
            # No valid suffix detected
            self.name = self.filename

        path = self._test_file(path)
        self.dim = (0, 0)

        tiles = glob.glob(os.path.join(path, '*' + ext))
        tiles = self._sort(tiles)  # calculates self.dim[0] if it can
        self.size = Image.open(tiles[0]).size

        try:
            params = pickle.load(open(param_file, 'rb'))
        except (IOError, TypeError):
            cprint('Set tilset parameters:')
            if not self.dim[0]:
                num_cols = int(prompt(' Number of columns:', '^\d+$'))
            else:
                num_cols = self.dim[0]
                cprint((' Number of columns: {} (determined from'
                        ' filenames)').format(num_cols))
            self.mag = float(prompt(' Magnification:', '^\d+(\.\d)?$'))
            self.snake = prompt(' Snake pattern?', {'y' : True, 'n' : False})
            review = True
        else:
            num_cols = params['num_cols']
            self.mag = params['mag']
            self.snake = params['snake']
            review = False

        skiplist = []
        if skip_file is not None:
            skiplist = self._handle_skipped(skip_file)
        tiles = self._patch(tiles, skiplist)

        self.grid = mandolin(tiles, num_cols)
        self.dim = (num_cols, len(self.grid))
        if self.snake:
            self.grid = self._desnake(self.grid, self.snake)

        # Try again if review fails
        if review:
            # Review parameters
            cprint('Review parameters for your mosaic:')
            cprint(' Dimensions:     {}x{}'.format(self.dim[0], self.dim[1]))
            cprint(' Magnification:  {}'.format(self.mag))
            cprint(' Snake :         {}'.format(self.snake))
            if review and not prompt('Confirm', {'y' : True, 'n' : False}):
                self.populate_tiles(path)
            else:
                self.keypoints = {}
                return self
        else:
            self.keypoints = {}
            return self




    def _test_file(self, path):
        """Test first tile in tileset to confirm PIL can open it

        If the image fails to open, this function will try
        to mogrify a usable copy in path/working. This requires
        ImageMagick.

        Args:
            path (str): filepath to tiles
            ext (str): extension on image files

        Returns:
            Path to set of images
        """
        for fp in glob.iglob(os.path.join(path, '*')):
            ext = os.path.splitext(fp)[1]
            try:
                Image.open(fp)
            except IOError:
                # This is a clumsy solution to PIL's unreliability
                # reading TIFFs. It uses ImageMagick to copy
                # the unreadable tiles to a subdirectory; the
                # IM-created tiles should always be readable by PIL.
                if ext in IMAGE_TYPES and mogrify(path, ext):
                    path = os.path.join(path, 'working')
                else:
                    cprint('Encountered unreadable tiles but could'
                           ' not fix them. Try installing ImageMagick'
                           ' and re-running this script.')
                    sys.exit()
            break
        return path, ext




    def _sort(self, tiles):
        """Identifies iterator in tileset and sorts

        The sort function works by detecting the iterator, which
        is the part of the filename that changes between files
        in the same tileset. Typically the interator will be an
        integer (abc-1.jpg or abc-001.jpg) or a column-row pair
        (abc_Grid[@0 0].jpg).

        Args:
            tiles (list): filepaths of all tiles as strings

        Returns:
            A sorted list of filepaths representing tiles, with
            empty strings where the tileset was patched.
        """
        # Identify this iterator by finding which parts change across
        # the tileset.
        starts_with = []
        ends_with = []
        i = 0
        while i < len(tiles):
            j = 0
            while tiles[i][j] == tiles[i-1][j]:
                j += 1
            starts_with.append(j)
            j = 0
            while tiles[i][::-1][j] == tiles[i-1][::-1][j]:
                j += 1
            ends_with.append(j)
            i += 1
        starts = tiles[0][:min(starts_with)]
        ends = tiles[0][len(tiles[0])-min(ends_with):]
        # Now we handle the two cases described above (number and
        # column-row pair). Note that the script is quite simple
        # in its handling of coordinates--for example, it does not
        # handle row-column pairs or column-row pairs joined by an "x."
        temp = {}
        cols = []
        e = None
        for tile in tiles:
            key = tile.replace(starts, '', 1).replace(ends, '', 1)
            try:
                # Special case: SEM grid notation. We can use the
                # coordinates in the grid to calculate the number
                # of columns.
                x, y = key.split(' ')
                cols.append(int(x))
                i = key
            except ValueError:
                try:
                    i = int(key)
                except ValueError:
                    # Typically caused by alien tiles in the tileset
                    e = ('Warning: Could not sort tiles. Please'
                         ' confirm that there are no extra tiles'
                         ' in the source folder.')
            temp[i] = tile
        cprint(e)
        if len(cols) and not self.dim[1]:
            num_cols = max(cols) + 1
            for key in temp.keys():
                x, y = key.split(' ')
                i = int(x) + num_cols * int(y)
                temp[i] = temp[key]
                del temp[key]
            self.dim = (num_cols, self.dim[1])
        # Create tiles and rows
        tiles = [temp[key] for key in sorted(temp.keys())]
        return tiles




    def _patch(self, tiles, skiplist=[]):
        """Patches tileset based on list of skipped tiles

        Args:
            tiles (list): list of tiles
            skiplist (list): list of indices (not filenames) from
                the master tileset that have been excluded by the
                user

        Returns:
            Sorted list of tiles with empty strings at indices
            where tiles were missing.
        """
        if not len(skiplist):
            return tiles
        else:
            # Get dimensions of the tileset from skipped file, then
            # check length of tiles against the number of skipped tiles
            # to see if anything's missing.
            dim = skiplist[0].split(': ', 1)[1].strip()
            self.dim =  tuple([int(x) for x in dim.split('x')])
            remainder = len(tiles) % self.dim[0]
            n = self.dim[0] * self.dim[1]
            if n == len(tiles) + (self.dim[0] - remainder):
                for i in skiplist[1:]:
                    tiles[i] = ''
                return tiles
        # Insert blanks where they should fall in the tile sequence,
        # then fill the tiles around them.
        sequence = {}
        for i in skiplist[1:]:
            sequence[i] = ''
        i = 0
        for tile in tiles:
            while True:
                try:
                    sequence[i]
                except KeyError:
                    sequence[i] = tile
                    break
                else:
                    i += 1
        tiles = []
        for i in sorted(sequence.keys()):
            tiles.append(sequence[i])
        cprint('Tile set was patched!')
        # Check for missing tiles
        if len(tiles) < n:
            print 'However, {} tiles appear to be missing'.format(n-len(tiles))
        return tiles




    def _missing(self, tiles):
        """Check tilset for missing tiles by analyzing keys

        Ars:
            tiles (dict): collection of tiles keyed to their index in
                the grid before desnaking

        Returns:
            None
        """
        keys = tiles.keys()
        idealized_keys = set([x for x in xrange(min(keys), max(keys)+1)])
        missing = idealized_keys - set(keys)
        empties = [tile for tile in tiles if not bool(tile)]
        if len(missing) and not len(empties):
            cprint('Warning: The following tiles appear to be missing:')
            for key in sorted(missing):
                cprint(' Index {}'.format(key))
                tiles.insert(key-1, '')
        else:
            cprint('All tiles appear to be present')




    def _desnake(self, grid, pattern):
        """Reorders tiles to account for snake pattern

        Args:
            grid (list): grid of tiles
            pattern (bool): specifies if snake pattern

        Returns:
            List containing the tile grid corrected for snaking
        """
        if pattern:
            return [grid[i][::-1] if i % 2 else grid[i]
                    for i in xrange(0, len(grid))]
        return tiles




    def _handle_skipped(self, skip_file):
        """Read indices of skipped tiles from file

        Args:
            skip_file (str): filepath to text file containing the
                list of skipped indices

        Returns:
            A list of indices used to patch the tileset
        """
        try:
            f = open(os.path.join(skip_file), 'rb')
        except OSError:
            raise OSError
        else:
            try:
                return [int(i.strip()) if i.isdigit() else i.strip()
                        for i in f.read().splitlines()]
            except TypeError:
                raise TypeError




    def prepare_mosaic(self, param_file=None, opencv=True, **kwargs):
        """Determines coordinates for tiles based on tileset metadata

        Args:
            param_file (str): filepath to parameters file`
            opencv (bool): specifies whether to use OpenCV. If
                OpenCV is not installed, will switch to manual.
            kwargs: see :py:func:`~Stitch2D.Mosaic.mosey`
                for additional keywords


        Returns:
            A dict of coordinates keyed to filepath
        """
        # Confirm that OpenCV is installed and working properly
        if opencv:
            try:
                cv2.imread(os.path.join(self.basepath, 'files', 'test.png'), 0)
            except NameError:
                cprint('Could not find OpenCV! Switching to manual stitch.')
                opencv = False

        try:
            params = pickle.load(open(param_file, 'rb'))
        except (IOError, TypeError):
            cprint(' Autostitch:     {}'.format(opencv))
            if opencv:
                cprint('Using OpenCV to stitch mosaic')
                defaults = {
                    'equalize_histogram' : False,
                    'matcher' : 'brute-force',
                    'scalar' : 0.5,
                    'threshold' : 0.7,
                }
                cv_params = {}
                for key in defaults:
                    try:
                        cv_params[key] = kwargs[key]
                    except KeyError:
                        cv_params[key] = defaults[key]
                cprint('  Equalize histogram: {}'.format(
                            cv_params['equalize_histogram']))
                cprint('  Matcher:            {}'.format(cv_params['matcher']))
                cprint('  Scalar:             {}'.format(cv_params['scalar']))
                cprint('  Threshold:          {}'.format(
                            cv_params['threshold']))
                cprint('Determining offset...')
                coordinates = self.cv_coordinates(**kwargs)
            else:
                cprint('Setting offset...')
                coordinates = self.set_coordinates()
            # Record job parameters to file
            params = [
                self.filename,
                '-' * len(self.filename),
                'Dimensions: {}x{}'.format(self.dim[0], self.dim[1]),
                'Magnification: {}'.format(self.mag),
                'Snake: {}'.format(self.snake),
                ''
            ]
            if opencv:
                params.extend([
                    'Autostitch: {}'.format(opencv),
                    'Equalize histogram: {}'.format(
                        cv_params['equalize_histogram']),
                    'Matcher: {}'.format(cv_params['matcher']),
                    'Scalar: {}'.format(cv_params['scalar']),
                    'Threshold: {}'.format(cv_params['threshold']),
                    ''
                    ])
            params.append('Tile coordinates:')
            keys = sorted(coordinates.keys(), key=lambda s:
                            'x'.join(['0'*(4-len(n))+n
                            for n in s.split('x')][::-1]))
            for key in keys:
                params.append('{}: {}'.format(key, coordinates[key]))
            fp = self.filename + '.txt'
            with open(fp, 'wb') as f:
                f.write('\n'.join(params))
            # Pickle key parameters for re-use later
            params = {
                'coordinates' : coordinates,
                'num_cols' : self.dim[0],
                'mag' : self.mag,
                'snake' : self.snake
            }
            with open(param_file, 'wb') as f:
                pickle.dump(params, f)
        else:
            cprint('Found parameters file')
            coordinates = params['coordinates']
        return coordinates




    def create_mosaic(self, coordinates, label=True, create_jpeg=True):
        """Draws mosaic based on the tile coordinates

        Args:
            coordinates (dict): normalized coordinates keyed to filepaths
            label (bool): specifies whether to include a label
                at the bottom of the final mosaic
            create_jpeg (bool): specifies whether to create a
                half-size JPEG derivative of the final mosaic

        Returns:
            None
        """
        # Normalize coordinates and calculate dimensions. The
        # dimensions of the mosaic are determined by the tile
        # dimensions minus the offsets between rows and columns
        # Some general notes:
        #  * Coordinates increase from (0,0) in the top left corner
        #  * Offsets are always applied as n - 1 because they occur
        #    between tiles.
        start_time = datetime.now()

        grid = self.grid
        w, h = self.size

        mosaic_width = max([coordinate[0] for coordinate in
                            coordinates.values()]) + w
        mosaic_height = max([coordinate[1] for coordinate in
                             coordinates.values()]) + h
        if label:
            label_height = int(mosaic_height * 0.04)
            mosaic_height += label_height
        cprint('Mosaic will be {:,} by {:,}'
               ' pixels'.format(mosaic_width, mosaic_height))
        # Create the mosaic
        cprint('Stitching mosaic...')
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255,255,255))
        n_row = 0
        while n_row < len(grid):
            row = grid[n_row]
            n_col = 0
            while n_col < len(row):
                position = 'x'.join([str(n) for n in [n_col, n_row]])
                try:
                    x, y = coordinates[position]
                except KeyError:
                    n_col += 1
                    continue
                else:
                    fp = grid[n_row][n_col]
                    path = os.path.dirname(fp)
                # Encode the name
                try:
                    mosaic.paste(Image.open(fp.encode('cp1252')), (x, y))
                except UnicodeDecodeError:
                    mosaic.paste(Image.open(fp), (x, y))
                except OSError:
                    cprint('Encountered unreadable tiles but'
                           ' could not fix them. Try installing'
                           ' ImageMagick and re-running this'
                           ' script.')
                    sys.exit()
                n_col += 1
            n_row += 1
        # Add label
        if label:
            ttf = os.path.join(self.basepath, 'files', 'OpenSans-Regular.ttf')
            text = self.name
            text = text[0].upper() + text[1:]
            draw = ImageDraw.Draw(mosaic)
            # Resize text to a reasonable size based on the
            # dimensions of the mosaic
            size = 100
            font = ImageFont.truetype(ttf, size)
            w, h = font.getsize(text)
            size = int(0.8 * size * label_height / float(h))
            font = ImageFont.truetype(ttf, size)
            x = int(0.02 * mosaic_width)
            y = mosaic_height - int(label_height)
            draw.text((x, y), text, (0, 0, 0), font=font)
        cprint('Saving as {}...'.format('TIFF'))
        #fp = os.path.join(self.path, os.pardir, self.filename + '.tif')
        fp = self.filename + '.tif'
        mosaic.save(fp, 'TIFF')
        if create_jpeg:
            cprint('Saving as JPEG...')
            fp = os.path.splitext(fp)[0] + '.jpg'
            try:
                mosaic = mosaic.resize((mosaic_width / 2, mosaic_height / 2))
            except:
                print 'Failed to resize JPEG. Creating full-size instead.'
                pass
            mosaic.save(fp, 'JPEG')
        cprint('Mosaic complete! (t={})'.format(datetime.now() - start_time))
        if path.rstrip('/').endswith('working'):
            try:
                shutil.rmtree(path)
            except OSError:
                pass
        return True




    def _set_coordinates(self):
        """Allows user to set offsets and coordinates using GUI

        Calls :py:func:`~Stitch2D.offset.OffsetEngine` to determine
        offset within and between rows. The same offset is used across
        the entire grid, so this will produce a useful but imperfect
        mosaic.

        Returns:
            A dict of coordinates specifying the placement of
            tiles on the final mosaic
        """
        grid = self.grid
        dim = self.grid
        w, h = self.size

        # Use the offset engine to allow users to set the offsets
        engine = OffsetEngine(grid, True)
        offset = engine.set_offset()
        engine = OffsetEngine(grid, False, offset)
        offset = engine.set_offset()

        x_offset_within_row = offset[0]
        y_offset_within_row = offset[1]
        x_offset_between_rows = offset[2]
        y_offset_between_rows = offset[3]

        coordinates = {}
        n_row = 0  # index of row
        for row in grid:
            n_col = 0  # index of column
            for fp in row:
                if bool(fp):
                    # Calculate x coordinate
                    x = ((w + x_offset_within_row) * n_col +
                         x_offset_between_rows * n_row)
                    if x_offset_between_rows < 0:
                        x -= x_offset_between_rows * (dim[1] - 1)
                    # Calculate y coordinate
                    y = ((h + y_offset_between_rows) * n_row +
                         y_offset_within_row * n_col)
                    if y_offset_within_row < 0:
                        y -= y_offset_within_row * (dim[0] - 1)
                    position = 'x'.join([str(n) for n in [n_col, n_row]])
                    coordinates[position] = (x, y)
                n_col += 1
            n_row += 1
        return self.normalize_coordinates(coordinates)




    def _cv_coordinates(self, detector='SIFT', **kwargs):
        """Uses OpenCV to determine coordinates of tiles in mosaic

        Tiles are placed by identifying a well-positioned tile and
        building outward. This approach performs poorly for objects
        with multiple features of interest separated by featureless
        expanses.

        Args
            detector (str): name of feature-detection algoritm.
                Currently, the only acceptable value is SIFT.
            kwargs: see :py:func:`~Stitch2D.Mosaic.mosey`
                for additional keywords

        Returns:
            A dict of coordinates specifying the placement of
            tiles on the final mosaic
        """
        tiles = {}
        n_row = 0
        while n_row < len(self.grid):
            row = self.grid[n_row]
            n_col = 0
            while n_col < len(row):
                tile = row[n_col]
                try:
                    tiles[tile]
                except KeyError:
                    tiles[tile] = {
                        'position' : (n_col, n_row),
                        'offsets' : [],
                        'coordinates' : None
                    }
                if n_col:
                    neighbor = row[n_col-1]
                    offset = self.cv_match(tile, neighbor, **kwargs)
                    if offset:
                        xy, n_total, n_cluster = offset
                        nxy = (xy[0]*-1, xy[1]*-1)
                        score = self.cv_reliability(n_cluster, n_total)
                        tiles[tile]['offsets'].append(['left', xy, score])
                        tiles[neighbor]['offsets'].append(['right', nxy,
                                                           score])
                if n_row:
                    neighbor = self.grid[n_row-1][n_col]
                    offset = self.cv_match(tile, neighbor, **kwargs)
                    if offset:
                        xy, n_total, n_cluster = offset
                        nxy = (xy[0]*-1, xy[1]*-1)
                        score = self.cv_reliability(n_cluster, n_total)
                        tiles[tile]['offsets'].append(['top', xy, score])
                        tiles[neighbor]['offsets'].append(['down', nxy,
                                                           score])
                n_col += 1
            n_row += 1

        self.analyze_offsets(tiles)

        # Score matches between tiles to find a well-positioned tile
        root = None
        high_score = -1
        for tile in sorted(tiles):
            offsets = tiles[tile]['offsets']
            try:
                score = sorted(offsets, key=lambda s:s[2]).pop()[2]
            except IndexError:
                pass
            else:
                if score > high_score:
                    root = tile
                    high_score = score

        # Place tiles relative to the root tile from above
        position = 'x'.join([str(n) for n in tiles[root]['position']])
        coordinates = {position : (0,0)}
        roots = [position]
        while True:
            neighbors = []
            for root in roots:
                n_col, n_row = [int(n) for n  in root.split('x')]
                tile = tiles[self.grid[n_row][n_col]]
                cprint(('Positioning tiles adjacent to'
                       ' {}...').format(os.path.basename(root)), self.verbose)
                offsets = tile['offsets']
                offsets = [offset for offset in offsets if offset[2] >= 5]
                for offset in offsets:
                    direction = offset[0]
                    dx, dy = offset[1]
                    if direction == 'top':
                        x, y = n_col, n_row-1
                    elif direction == 'right':
                        x, y = n_col+1, n_row
                    elif direction == 'down':
                        x, y = n_col, n_row+1
                    elif direction == 'left':
                        x, y = n_col-1, n_row
                    try:
                        self.grid[y][x]
                    except IndexError:
                        pass
                    else:
                        position = 'x'.join([str(n) for n in [x, y]])
                        try:
                            coordinates[position]
                        except KeyError:
                            x, y = coordinates[root]
                            coordinates[position] = (x + dx, y + dy)
                            neighbors.append(position)
            roots = neighbors
            if not len(roots):
                break
        return self.normalize_coordinates(coordinates)




    def _cv_match(self, img1, img2, detector='SIFT', **kwargs):
        """Use OpenCV to match features between two images

        Args:
            img1 (str): path to image
            img2 (str): path to another image
            detector: the name of an OpenCV feature detector,
                like SIFT or ORB. Currently only SIFT works.
            kwargs: see :py:func:`~Stitch2D.Mosaic.mosey`
                for additional keywords

        Returns:
            (x, y), n_clsuter, n_total

            Includes average offset, the number of features in the
            largest cluster, and the total number of features detected.
            Returns None on failure.
        """
        cprint('OpenCV: cv2.xfeatures2d.SIFT_create()', self.verbose)
        detector = cv2.xfeatures2d.SIFT_create()

        start_time = datetime.now()

        fn1 = os.path.basename(img1)
        fn2 = os.path.basename(img2)

        # Read descriptors
        for fp in (img2, img1):
            if not bool(fp):
                return None
            try:
                self.keypoints[fp]
            except KeyError:
                fn = os.path.basename(fp)
                cprint('Getting keypoints for {}'.format(fn), self.normal)
                cprint('OpenCV: cv2.imread()', self.verbose)
                img = cv2.imread(fp, 0)
                if kwargs['scalar'] < 1:
                    h, w = [int(n*kwargs['scalar']) for n in img.shape]
                    cprint('OpenCV: cv2.resize()', self.verbose)
                    img = cv2.resize(img, (w, h))
                if kwargs['equalize_histogram']:
                    cprint('OpenCV: cv2.equalizeHist()', self.verbose)
                    img = cv2.equalizeHist(img)
                cprint('OpenCV: cv2.detectAndCompute()', self.verbose)
                self.keypoints[fp] = detector.detectAndCompute(img, None)

        kp1, des1 = self.keypoints[img1]
        if not (any(kp1) and np.any(des1)):
            cprint('No keypoints found in {}'.format(fn1), self.normal)
            return None

        kp2, des2 = self.keypoints[img2]
        if not (any(kp2) and np.any(des2)):
            cprint('No keypoints found in {}'.format(fn2), self.normal)
            return None

        if kwargs['matcher'] == 'flann':
            # FLANN-based matching. Segfaults in Ubuntu :(.
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            cprint('OpenCV: cv2.FlannBasedMatcher()', self.verbose)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                cprint('OpenCV: cv2.FlannBasedMatcher.knnMatch()', self.verbose)
                matches = flann.knnMatch(des1, des2, k=2)
            except cv2.error:
                cprint('No matches found in'
                       '{1} and {0}'.format(fn1, fn2), self.normal)
                return None
        else:
            # Brute force matching. Slower.
            cprint('OpenCV: cv2.BFMatcher()', self.verbose)
            bf = cv2.BFMatcher()
            try:
                cprint('OpenCV: cv2.BFMatcher.knnMatch()', self.verbose)
                matches = bf.knnMatch(des1, des2, k=2)
            except cv2.error:
                cprint('No matches found in'
                       '{1} and {0}'.format(fn1, fn2), self.normal)
                return None

        # Matches consist of DMatch objects, which among other things
        # contain coordinates for keypoints in kp1 and kp2. These can
        # be used to calculate an average offset between two tiles;
        # the average is based on a simple cluster analysis of matches
        # detected between the two images.

        # Identify good matches using ratio test from Lowe's paper. The
        # length test bypasses an error in which some matches returned
        # by the detector.knnMatch() function have <k results.
        good = []
        for m, n in [m for m in matches if len(m)==2]:
            if m.distance < kwargs['threshold'] * n.distance:
                good.append(m)

        fn1 = os.path.basename(img1)
        fn2 = os.path.basename(img2)
        x = []
        y = []

        '''
        # Identify inliers using homography
        if len(good) >= 5:
            src_pts = np.float32([kp1[m.queryIdx].pt
                                  for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt
                                  for m in good]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = mask.ravel().tolist()

            # Use inlier list to determine offset
            i = 0
            while i < len(matchesMask):
                c1 = good[i].queryIdx
                c2 = good[i].trainIdx
                x.append((kp1[c1].pt[0] - kp2[c2].pt[0]) / scalar)
                y.append((kp1[c1].pt[1] - kp2[c2].pt[1]) / scalar)
                i += 1
            x_avg = sum(x) / len(x)
            y_avg = sum(y) / len(y)
            print fill(('Matched {} features in {} and {}'
                        ' (t={})').format(len(x), fn1, fn2, dt),
                       subsequent_indent='  ')
            return (x_avg, y_avg), n, m
        '''

        # Identify inliers using a simple clustering approach
        for m in good:
            c1 = m.queryIdx
            c2 = m.trainIdx
            x.append((kp1[c1].pt[0] - kp2[c2].pt[0]) / kwargs['scalar'])
            y.append((kp1[c1].pt[1] - kp2[c2].pt[1]) / kwargs['scalar'])
        if len(x) and len(y):
            groups = cluster(x, 2)
            x_max = max([len(group) for group in groups])
            group = [group for group in groups if len(group)==x_max][0]
            x_avg = sum(group) / len(group)

            groups = cluster(y, 2)
            y_max = max([len(group) for group in groups])
            group = [group for group in groups if len(group)==y_max][0]
            y_avg = sum(group) / len(group)

            # Return coordinates, total size, and cluster size
            n = len(x)
            m = min([x_max, y_max])
            dt = datetime.now() - start_time
            cprint(('Matched {}/{} features in {} and {}'
                    ' (t={})').format(m, n, fn1, fn2, dt), self.normal)
            return (x_avg, y_avg), n, m
        return None




    def _cv_reliability(self, n_cluster, n_total):
        """Assess reliability of offset

        Args:
            n_cluster (int): size of largest cluster of features
            n_total (int): size of list of all features

        Returns:
            Reliability score, which is just the number of
            matches adjusted by frequency. Values above 5
            are considered reliable. This is not great.
        """
        # Offset reliability test
        #  1. Are there a large number of matches?
        #  2. Do most of these matches cluster in one group?
        #  Minima: 4/5, 4/6, 5/7, 5/8, 6/9, then >50% for 10+
        pct_cluster = n_cluster / float(n_total)
        return n_cluster * pct_cluster




    def _analyze_offsets(self, tiles):
        """Placeholder for future test"""
        for tile in tiles:
            tile = tiles[tile]
            n_row, n_col = tile['position']
            offsets = tile['offsets']




    def _normalize_coordinates(self, coordinates):
        """Calibrates calculated coordinates to zero the minimum on each axis

        Args:
            coordinates (dict): raw coordinates for each tile in mosaic

        Returns:
            A dict containing the calibrated coordinates for each
            tile.
        """
        x_min = min([coordinate[0] for coordinate in coordinates.values()])
        x_max = max([coordinate[0] for coordinate in coordinates.values()])
        y_min = min([coordinate[1] for coordinate in coordinates.values()])
        y_max = max([coordinate[1] for coordinate in coordinates.values()])

        # Shift everything so that minimum coordinates are 0,0
        for tile in coordinates:
            x, y = coordinates[tile]
            coordinates[tile] = int(x - x_min), int(y - y_min)

        return coordinates




def mosey(path=None, param_file='params.p', skip_file=None,
          create_jpeg=False, opencv=True, **kwargs):
    """Stitches a set of directories using one set of parameters

    Use this function to stitch derivatives from a master file
    (like NSS element maps) or images collected at the same time
    under different light source.

    If no list of skipped tiles is provided, mosey will check
    for one along the specified path.

    Args:
        path (str): filepath to either a directory containing tiles
            *or* a directory containing one or more directories
            containing tiles
        param_file (str): filepath to parameters file
        skip_file (str): filepath to a list of skipped tiles. If
            excluded, the function will search the path for a list.
        opencv (bool): specifies whether to use OpenCV. If
            OpenCV is not installed, the function will revert
            to matching tiles manually.
        create_jpeg (bool): specifies whether to create a
            half-size JPEG derivative of the final mosaic
        equalize_histogram (bool): specifies whether to equalize
            histogram before matching features (\*\*kwarg)
        matcher (str): name of feature-matching algoritm (\*\*kwarg)
        scalar (float): amount to scale imahes before matching
            (\*\*kwarg)
        threshold (float): threshold for Lowe test (\*\*kwarg)

    Returns:
        None
    """
    if not path:
        root = Tkinter.Tk()
        root.withdraw()
        title = 'Please select the directory containing your tile sets:'
        initial = os.path.expanduser('~')
        kwargs['path'] = tkFileDialog.askdirectory(parent=root, title=title,
                                                   initialdir=initial)
    # Check for tiles. If none found, try the parent directory.
    try:
        tilesets = [os.path.join(path, dn) for dn in os.listdir(path)
                    if os.path.isdir(os.path.join(path, dn))
                    and not dn == 'skipped']
    except:
        cprint('Invalid path : {}. Exiting.'.format(path))
        sys.exit()
    if not len(tilesets):
        cprint('No subdirectories found in {}. Processing'
               ' main directory instead.'.format(path))
        tilesets = [path]
    for path in copy(tilesets):
        if 'bsed' in path or 'ppl' in path:
            tilesets.insert(0, tilesets.pop(tilesets.index(path)))
    # Check for skipped files. By default, mosey will check all
    # subdirectories of the main directory for skipped file list and then
    # apply that list to everything processed in the current job. Also,
    # element maps can be hit or miss for setting offsets, so we shift
    # backscatter images to top of the list if they're available.
    if skip_file is None:
        skip_files = []
        for path in tilesets:
            try:
                open(os.path.join(path, 'skipped.txt'))
            except IOError:
                pass
            else:
                skip_files.append(path)
        if len(skip_files) > 1:
            cprint('Warning: Multiple skip lists found:\n ' +
                   ' \n'.join(skip_files))
        if len(skip_files):
            skip_file = skip_file[0]
            cprint('Using list of skipped tiles'
                   ' from {}'.format(skip_file))
    coordinates = None
    for path in tilesets:
        cprint('New tileset: {}'.format(os.path.basename(path)))
        mosaic = Mosaic(path, param_file, skip_file)
        if not coordinates:
            coordinates = mosaic.prepare_mosaic(param_file, opencv, **kwargs)
        mosaic.create_mosaic(coordinates, create_jpeg=create_jpeg)
    # Remove parameters file
    try:
        os.remove(param_file)
    except OSError:
        pass
