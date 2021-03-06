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
import re
import shlex
import shutil
import subprocess
import sys
import time
import json as serialize
from copy import copy
from datetime import datetime
from textwrap import fill

try:
    import cv2
except ImportError:
    pass
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .helpers import (
    blur,
    brighten,
    cluster,
    cprint,
    mandolin,
    mogrify,
    prompt,
    read_image,
     _guess_extension,
     _select_folder,
     IMAGE_MAP
)
from .offset import OffsetEngine




IMAGE_TYPES = {
    'unspecified' : 'unspecified image type',
    'ref' : 'petrographic microscope, reflected light',
    'rfl' : 'petrographic microscope, reflected light',
    'rl' : 'petrographic microscope, reflected light',
    'ppl' : 'petrographic microscope, transmitted light',
    'trans' : 'petrographic microscope, transmitted light',
    'xpl' : 'petrographic microscope, cross-polarized light',
    'xpol' : 'petrographic microscope, cross-polarized light',
    'bse' : 'SEM, backscatter',
    'bsed' : 'SEM, backscatter',
    'nbsed' : 'SEM, normalized backscatter',
    'etd' : 'SEM, secondary electron',
    'sed' : 'SEM, secondary electron',
    'cl' : 'cathodoluminescence',
    'al' : 'SEM x-ray map, Al',
    'ca' : 'SEM x-ray map, Ca',
    'cr' : 'SEM x-ray map, Cr',
    'fe' : 'SEM x-ray map, Fe',
    'k' : 'SEM x-ray map, K',
    'mg' : 'SEM x-ray map, Mg',
    'mn' : 'SEM x-ray map, Mn',
    'na' : 'SEM x-ray map, Na',
    'ni' : 'SEM x-ray map, Ni',
    'o' : 'SEM x-ray map, O',
    's' : 'SEM x-ray map, S',
    'si' : 'SEM x-ray map, Si',
    'ti' : 'SEM x-ray map, Ti'
}




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
        fill (tuple): fill color of background. Default is black.
        text (tuple): color of text. Default is inverse of fill.
    """

    def __init__(self, path, output='.', param_file=None, skip_file=None,
                 label=None, **kwargs):
        """Initialize new Tileset

        The heavy lifting is done by
        :py:func:`~Stitch2d.Mosaic.populate_tiles()`,
        which populates and processes the tileset.

        Args:
            path (str): path to tileset
            param_file (str): filepath to parameters file
            skip_file (str): path to file containing indices of skipped tiles
        """
        self.basepath = os.path.dirname(__file__)
        try:
            os.makedirs(output)
        except OSError:
            pass
        self.output = output
        self.normal = True
        self.verbose = False
        self.fill = (0,0,0)
        self.text = tuple([(255 - x) for x in self.fill])
        self.grid = {}
        self.set_filename(path, label=label)
        # Does mosaic already exist?
        try:
            fp = os.path.join(self.output, self.filename + '.tif')
            open(fp, 'r')
            print('{} already exists'.format(fp))
        except FileNotFoundError:
            self.populate_tiles(path, param_file, skip_file, label, **kwargs)


    def set_filename(self, path, label=None):
        # Get descriptive name of tileset based on filename
        fn = os.path.splitext(os.path.basename(path))[0]
        name = label if label is not None else fn
        try:
            base, kind = fn.rsplit('_', 1)
        except ValueError:
            kind = None
            blurb = None
        else:
            while kind and kind[0].isdigit():
                kind = kind[1:]
            blurb = IMAGE_TYPES.get(kind.lower())
            if blurb is None and 1 <= len(kind) <= 2:
                blurb = kind.lower().capitalize()
        fn = name.replace(' ', '_') if label is not None else fn
        if kind is not None and not fn.endswith(kind):
            fn += '_{}'.format(kind)
        if blurb is not None:
            name += ' ({})'.format(blurb)
        self.filename = fn.replace('(', '').replace(')', '').replace('.', 'pt')
        self.name = name
        cprint('{} => {}'.format(self.name, self.filename), self.normal)



    def populate_tiles(self, path, param_file=None, skip_file=None,
                       label=None, **kwargs):
        """Test, characterize, sort and patch tiles from path

        Args:
            path (str): filepath to tiles
            param_file (str): filepath to parameters file
            skip_file (str): filepath to text file containing the
                list of skipped indices
            label (str): name of the mosaic (typically the sample name)
            num_columns (int): number of columns in the mosaic
            snake (bool): specifies whether mosaic is a snake pattern
            smooth (bool): specifies whether to try to smooth out boundaries
                between images

        Returns:
            None
        """
        # Get extension
        try:
            ext = _guess_extension(path)
        except KeyError:
            if param_file is None:
                raise
            else:
                return self

        path = self._test_file(path)
        self.dim = (0, 0)
        tiles = glob.glob(os.path.join(path, '*' + ext))
        try:
            tiles = self._sort(tiles)  # calculates self.dim[0] if it can
        except IndexError:
            pass
        self.count = len(tiles)
        print('The tileset contains {} tiles'.format(len(tiles)))
        # Set self.size to the LARGEST tile size. If multiple sizes are
        # present, resize and manual options are forbidden.
        sizes = [read_image(fp).size for fp in tiles[:5]]
        self.size = (max([size[0] for size in sizes]),
                     max([size[1] for size in sizes]))
        if len(set(sizes)) > 1:
            print('Tiles are not uniform in size!')

        try:
            params = serialize.load(open(param_file, 'r'))
            # Are the tiles the correct size?
            if (params['size'] != list(self.size)
                or params['count'] != self.count):
                    os.remove(param_file)
                    raise IOError('Params do not match current tileset')
        except (IOError, TypeError):
            # Get parameters from kwarg
            minval = kwargs.get('minval')
            num_cols = kwargs.get('num_cols')
            snake = kwargs.get('snake')
            smooth = kwargs.get('smooth')
            blur = kwargs.get('blur', 0)
            # Prompt for missing params and assign to attributes
            review = num_cols is None or snake is None
            cprint('Set tileset parameters:')
            if not self.dim[0] and num_cols is None:
                num_cols = int(prompt(' Number of columns:', '^\d+$'))
            elif num_cols is None:
                review = snake is None
                num_cols = self.dim[0]
                cprint((' Number of columns: {} (determined from'
                        ' filenames)').format(num_cols))
            #self.mag = float(prompt(' Magnification:', '^\d+(\.\d)?$'))
            if snake is None:
                snake = prompt(' Snake pattern?', {'y' : True, 'n' : False})
            self.snake = snake
            self.smooth = True if smooth else False
            self.minval = minval if minval is not None else 0
            self.blur = blur
        else:
            review = False
            num_cols = params['num_cols']
            self.minval = params['minval']
            self.snake = params['snake']
            self.smooth = params['smooth']
            self.blur = params['blur']

        skiplist = []
        if skip_file is not None:
            skiplist = self._handle_skipped(skip_file)
        tiles = self._patch(tiles, skiplist)

        # Pad the grid
        rows = {}
        for tile in tiles:
            from .helpers import _get_coordinates
            try:
                row, col = _get_coordinates(tile)
            except IndexError:
                row = len(rows)
                if len(rows.get(row, [])) == num_cols:
                    row += 1
            rows.setdefault(row, []).append(tile)
        max_tiles = max([len(row) for row in list(rows.values())])
        longest_row = [len(row) for i, row in rows.items() if len(row) == max_tiles][0]
        longest_row = 0
        for row, cols in rows.items():
            while len(cols) < longest_row:
                cols.append(None)
        #print longest_row
        #raw_input()

        self.grid = mandolin(tiles, num_cols)
        self.dim = (num_cols, len(self.grid))
        if self.snake:
            self.grid = self._desnake(self.grid, self.snake)

        # Review parameters, allowing user to try again if the parameters
        # are not suitable
        cprint('Mosaic parameters:')
        cprint(' Dimensions:     {}x{}'.format(self.dim[0], self.dim[1]))
        #cprint(' Magnification:  {}'.format(self.mag))
        cprint(' Snake:          {}'.format(self.snake))
        cprint(' Smooth:         {}'.format(self.smooth))
        cprint(' Blur radius:    {}'.format(self.blur))
        cprint(' Minimum pixel:  {}'.format(self.minval))
        if review and not prompt('Confirm', {'y' : True, 'n' : False}):
            self.populate_tiles(path, ext, param_file, skip_file, label)
        else:
            self.keypoints = {}
            return self


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
            params = serialize.load(open(param_file, 'r'))
            # Are the tiles the correct size?
            if (params['size'] != list(self.size)
                or params['count'] != self.count):
                    os.remove(param_file)
                    raise IOError('Params do not match current tileset')
        except (IOError, TypeError) as e:
            if opencv:
                cprint('Using OpenCV to stitch mosaic')
                defaults = {
                    'equalize_histogram' : False,
                    'matcher' : 'brute-force',
                    'homography' : False,
                    'scalar' : 0.5,
                    'threshold' : 0.7,
                }
                cv_params = {}
                for key in defaults:
                    cv_params[key] = kwargs.get(key, defaults[key])
                cprint('  Equalize histogram: {}'.format(
                            cv_params['equalize_histogram']))
                cprint('  Matcher:            {}'.format(cv_params['matcher']))
                cprint('  Homography:         {}'.format(
                            cv_params['homography']))
                cprint('  Scalar:             {}'.format(cv_params['scalar']))
                cprint('  Threshold:          {}'.format(
                            cv_params['threshold']))
                cprint('Determining offset...')
                posdata = self._cv_coordinates(**cv_params)
            else:
                cprint('Setting offset...')
                posdata = self._set_coordinates()
            # Record job parameters to file
            params = [
                self.filename,
                '-' * len(self.filename),
                'Dimensions: {}x{}'.format(self.dim[0], self.dim[1]),
                #'Magnification: {}'.format(self.mag),
                'Snake: {}'.format(self.snake),
                'Smooth: {}'.format(self.smooth),
                'Blur radius: {}'.format(self.blur),
                'Minimum pixel: {}'.format(self.minval),
                ''
            ]
            if opencv:
                params.extend([
                    'Autostitch: {}'.format(opencv),
                    'Equalize histogram: {}'.format(
                        cv_params['equalize_histogram']),
                    'Matcher: {}'.format(cv_params['matcher']),
                    'Homography: {}'.format(cv_params['homography']),
                    'Scalar: {}'.format(cv_params['scalar']),
                    'Threshold: {}'.format(cv_params['threshold']),
                    ''
                    ])
            coordinates = posdata['coordinates']
            params.append('Tile coordinates:')
            keys = sorted(list(coordinates.keys()), key=lambda s:
                            'x'.join(['0'*(4-len(n))+n
                            for n in s.split('x')][::-1]))
            for key in keys:
                params.append('{}: {}'.format(key, coordinates[key]))
            fp = os.path.join(self.output, self.filename + '.txt')
            with open(fp, 'w') as f:
                f.write('\n'.join(params))
            # Pickle key parameters for re-use later
            params = {
                'posdata' : posdata,
                'minval': self.minval,
                'num_cols': self.dim[0],
                #'mag' : self.mag,
                'snake': self.snake,
                'smooth': self.smooth,
                'blur': self.blur,
                'size': self.size,
                'count': self.count
            }
            with open(param_file, 'w') as f:
                serialize.dump(params, f)
        else:
            cprint('Found parameters file')
            posdata = params['posdata']
        return posdata


    def create_mosaic(self, posdata, label=True, create_jpeg=True):
        """Draws mosaic based on the tile coordinates

        Args:
            posdata (dict): positional data for tiles
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

        coordinates = posdata['coordinates']
        overlaps = posdata['overlaps']

        grid = self.grid
        w, h = self.size

        mosaic_width = max([coordinate[0] for coordinate in
                            list(coordinates.values())]) + w
        mosaic_height = max([coordinate[1] for coordinate in
                             list(coordinates.values())]) + h
        if label:
            label_height = int(mosaic_height * 0.04)
            mosaic_height += label_height
        cprint('Mosaic will be {:,} by {:,}'
               ' pixels'.format(mosaic_width, mosaic_height))
        # Create the mosaic
        cprint('Stitching mosaic...')
        # Group the tiles by size
        tiles = []
        n_row = 0
        while n_row < len(grid):
            row = grid[n_row]
            n_col = 0
            while n_col < len(row):
                position = '{}x{}'.format(n_col, n_row)
                try:
                    x, y = coordinates[position]
                except KeyError:
                    pass
                else:
                    fp = grid[n_row][n_col]
                    path = os.path.dirname(fp)
                    # When stitching multiple mosaics with the same params, you
                    # can sometimes run into tilesets with fewer tiles than the
                    # initial mosaic. Catch that error here.
                    try:
                        size = read_image(fp).size
                    except AttributeError:
                        pass
                    else:
                        area = size[0] * size[1]
                        tiles.append([area, fp, (x, y)])
                n_col += 1
            n_row += 1
        # Create a lookup from overlaps
        lookup = {}
        for t1, t2 in overlaps:
            data = {tuple(t1[0]): t1[1], tuple(t2[0]): t2[1]}
            for key in data:
                lookup.setdefault(key, []).append(data)

        # Normalize colors by comparing overlaps between adjacent tiles,
        # building out from the middle of the mosaic to minimize edge effects
        scalars = {}
        brightness = {}
        if self.smooth:
            found = []
            n_row = len(grid) // 2
            n_col = len(row) // 2
            roots = [(n_col, n_row)]  # pos in overlap is like this I guess
            while True:
                neighbors = []
                for root in roots:
                    found.append(root)
                    n_col, n_row = root
                    fp_root = grid[n_row][n_col]
                    scalars.setdefault(fp_root, 1.)

                    # Scalar computation does not work with at least mode=L,
                    # so force images to RGB for this part
                    im_root = read_image(fp_root, "RGB")
                    if self.blur:
                        im_root = blur(im_root, self.blur)

                    # Scale each tile to its neighbor
                    for neighbor in lookup.get(root, []):
                        coords = [c for c in neighbor.keys() if c != root][0]
                        dim1 = neighbor[root]
                        dim2 = neighbor[coords]
                        n_col, n_row = coords
                        try:
                            fp2 = grid[n_row][n_col]
                        except (IndexError, KeyError):
                            pass
                        else:
                            # Crop root to overlap with neighbor
                            im1 = im_root.crop(dim1)
                            arr1 = np.array(im1)

                            # Crop neighbor to overlap with root
                            im2 = read_image(fp2, "RGB").crop(dim2)
                            if self.blur:
                                im2 = blur(im2, self.blur)
                            arr2 = np.array(im2)

                            # Take mean of overlap for both images and
                            # calculate scalar needed to match neighbor
                            # to root
                            m1 = arr1[arr1 > 0].mean() * scalars.get(fp_root, 1.)
                            m2 = arr2[arr2 > 0].mean() * scalars.get(fp2, 1.)
                            scalars.setdefault(fp2, m1 / m2)

                            if not coords in found:
                                neighbors.append(coords)
                                brightness[fp2] = arr2[arr2 > 0].mean()

                roots = list(set(neighbors))
                if not roots:
                    break

        # Normalize scalars to brightest tile
        if scalars:
            brightest = sorted(brightness.items(), key=lambda kv: kv[1])[-1][0]
            base_scalar = scalars[brightest]
            scalars = {k: v / base_scalar for k, v in scalars.items()}

        # Paste tiles. If tiles are not uniform in size, paste them in
        # order of increasing size. This is intended to resolve an issue
        # with artifacts when using cropped images.
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), self.fill)
        for tile in tiles[::-1]:
            area, fp, coordinates = tile
            try:
                im = read_image(fp.encode('cp1252'))
            except (TypeError, UnicodeDecodeError):
                im = read_image(fp)
            except OSError:
                cprint('Encountered unreadable tiles but'
                       ' could not fix them. Try installing'
                       ' ImageMagick and re-running this'
                       ' script.')
                sys.exit()
            im = im.convert('RGB')

            if self.blur:
                im = blur(im, self.blur)

            if self.smooth or self.minval:
                data = np.array(im).astype(np.float64)
                if self.smooth:
                    data[data > 0] *= scalars.get(fp, 1.)
                if self.minval:
                    data[data < self.minval] = 0
                data[data > 255] = 255
                im = Image.fromarray(data.astype(np.uint8))
            mosaic.paste(im, coordinates)

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
            draw.text((x, y), text, self.text, font=font)

        cprint('Saving as {}...'.format('TIFF'))
        #fp = os.path.join(self.path, os.pardir, self.filename + '.tif')
        fp = os.path.join(self.output, self.filename + '.tif')
        mosaic.save(fp, 'TIFF')
        if create_jpeg:
            cprint('Saving as JPEG...')
            fp = os.path.splitext(fp)[0] + '.jpg'
            try:
                mosaic = mosaic.resize((mosaic_width // 2, mosaic_height // 2))
            except:
                print('Failed to resize JPEG. Creating full-size instead.')
                pass
            mosaic.save(fp, 'JPEG', quality=92)
        cprint('Mosaic complete! (t={})'.format(datetime.now() - start_time))
        if path.rstrip('/').endswith('working'):
            try:
                shutil.rmtree(path)
            except OSError:
                pass
        return True


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
                if ext in IMAGE_MAP and mogrify(path, ext):
                    path = os.path.join(path, 'working')
                else:
                    cprint('Encountered unreadable tiles but could'
                           ' not fix them. Try installing ImageMagick'
                           ' and re-running this script.')
                    sys.exit()
            break
        return path


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
                    i = key
                    # Typically caused by alien tiles in the tileset
                    #e = ('Warning: Could not sort tiles. Please'
                    #     ' confirm that there are no extra tiles'
                    #     ' in the source folder.')
            temp[i] = tile
        cprint(e)
        if len(cols) and not self.dim[1]:
            num_cols = max(cols) + 1
            for key in list(temp.keys()):
                x, y = key.split(' ')
                i = int(x) + num_cols * int(y)
                temp[i] = temp[key]
                del temp[key]
            self.dim = (num_cols, self.dim[1])
        # Create tiles and rows
        tiles = [temp[key] for key in sorted(temp.keys())]
        #tiles.sort()
        #for tile in tiles: print tile
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
            self.dim = tuple([int(x) for x in dim.split('x')])
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
            print('However, {} tiles appear to be missing'.format(n-len(tiles)))
        return tiles


    def _missing(self, tiles):
        """Check tilset for missing tiles by analyzing keys

        Ars:
            tiles (dict): collection of tiles keyed to their index in
                the grid before desnaking

        Returns:
            None
        """
        keys = list(tiles.keys())
        idealized_keys = set([x for x in range(min(keys), max(keys)+1)])
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
                    for i in range(0, len(grid))]
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
            f = open(os.path.join(skip_file), 'r')
        except OSError:
            raise OSError
        else:
            try:
                return [int(i.strip()) if i.isdigit() else i.strip()
                        for i in f.read().splitlines()]
            except TypeError:
                raise TypeError


    def _set_coordinates(self):
        """Allows user to set offsets and coordinates using GUI

        Calls :py:func:`~Stitch2D.offset.OffsetEngine` to determine
        offset within and between rows. The same offset is used across
        the entire grid, so this will produce an imperfect mosaic.

        Returns:
            A dict of coordinates specifying the placement of
            tiles on the final mosaic
        """
        grid = self.grid
        dim = self.dim
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

        tiles = {}
        coordinates = {}
        overlaps = []
        n_row = 0  # index of row
        for row in grid:
            row = self.grid[n_row]
            n_col = 0  # index of column
            for fp in row:
                tile = row[n_col]
                pos = (n_col, n_row)
                if fp:
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
                    # Calculate overlaps for smoothing calculation
                    if n_row and self.smooth:
                        xy = (x_offset_between_rows, y_offset_between_rows)
                        neighbor = self.grid[n_row - 1][n_col]
                        if tile and neighbor:
                            overlaps.append(
                                self._row_compare(xy, pos, tile, neighbor, True)
                            )
                    if n_col and self.smooth:
                        xy = (x_offset_within_row, y_offset_within_row)
                        neighbor = row[n_col - 1]
                        if tile and neighbor:
                            overlaps.append(
                                self._col_compare(xy, pos, tile, neighbor, True)
                            )
                n_col += 1
            n_row += 1
        self.overlaps = overlaps
        return self._normalize_coordinates(coordinates)


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
        overlaps = []
        n_row = 0
        while n_row < len(self.grid):
            row = self.grid[n_row]
            n_col = 0
            while n_col < len(row):
                tile = row[n_col]
                pos = (n_col, n_row)
                try:
                    tiles[tile]
                except KeyError:
                    tiles[tile] = {
                        'position' : pos,
                        'offsets' : [],
                        'coordinates' : None
                    }
                if n_col:
                    neighbor = row[n_col-1]
                    offset = self._cv_match(tile, neighbor, **kwargs)
                    if offset:
                        xy, n_total, n_cluster = offset
                        nxy = (xy[0]*-1, xy[1]*-1)
                        score = self._cv_reliability(n_cluster, n_total)
                        tiles[tile]['offsets'].append(['left', xy, score])
                        tiles[neighbor]['offsets'].append(['right',
                                                           nxy,
                                                           score])
                        if score > 5 and self.smooth:
                            overlaps.append(
                                self._col_compare(xy, pos, tile, neighbor)
                            )
                if n_row:
                    neighbor = self.grid[n_row-1][n_col]
                    offset = self._cv_match(tile, neighbor, **kwargs)
                    if offset:
                        #raw_input(offset)
                        xy, n_total, n_cluster = offset
                        nxy = (xy[0]*-1, xy[1]*-1)
                        score = self._cv_reliability(n_cluster, n_total)
                        tiles[tile]['offsets'].append(['top', xy, score])
                        tiles[neighbor]['offsets'].append(['down',
                                                           nxy,
                                                           score])
                        if score > 5 and self.smooth:
                            overlaps.append(
                                self._row_compare(xy, pos, tile, neighbor)
                            )
                n_col += 1
            n_row += 1
        self.overlaps = overlaps
        self._analyze_offsets(tiles)
        '''
        # Score matches between tiles to find a well-positioned tile
        # in the middle 50% of image
        root = None
        high_score = -1
        min_col = 0 * n_col
        max_col = n_col - min_col
        min_row = 0 * n_row
        max_row = n_col - min_row
        for tile in sorted(tiles):
            n_col, n_row = tiles[tile]['position']
            if (min_col < n_col < max_col and min_row < n_row < max_row):
                offsets = tiles[tile]['offsets']
                try:
                    score = sorted(offsets, key=lambda s:s[2]).pop()[2]
                except IndexError:
                    pass
                else:
                    #print '{}x{}: {}'.format(n_col, n_row, score)
                    if score > high_score:
                        root = tile
                        high_score = score
        '''
        islands = {}
        for tile in [tile for tile in tiles if tile]:
            key = copy(tile)
            root = tile
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
                           ' {}...').format(os.path.basename(root)),
                                            self.verbose)
                    offsets = tile['offsets']
                    offsets = [offset for offset in offsets if offset[2] >= 5]
                    for offset in offsets:
                        direction = offset[0]
                        dx, dy = offset[1]
                        if direction == 'top':
                            x, y = n_col, n_row - 1
                        elif direction == 'right':
                            x, y = n_col+1, n_row
                        elif direction == 'down':
                            x, y = n_col, n_row + 1
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
            islands[key] = coordinates
        # Identify largest island
        n = max([len(islands[key]) for key in islands])
        for key in sorted(islands):
            if len(islands[key]) == n:
                coordinates = islands[key]
                break
        cprint('{} selected as root ({}/{} tiles represented)'.format(
                    os.path.basename(key), n, len(tiles)))
        return self._normalize_coordinates(coordinates)


    def _row_compare(self, xy, tile, fp_lower, fp_upper, manual=False):
        """Compares overlapping sections of tiles in adjacent rows

        Args:
            xy (list): offsets as (dx, dy)
            tile (list): (row, col) of primary image
            fp_lower: path to primary (lower) image
            fp_upper: path to neighbor above
        """
        # Load images
        im_lower = read_image(fp_lower)
        im_upper = read_image(fp_upper)
        if im_lower.size != im_upper.size:
            raise ValueError("images are different sizes")
        w, h = im_lower.size

        # Offsets set manually differ from those calculated by OpenCV.
        # Standardize the offsets here to the OpenCV way.
        dx, dy = [int(n) for n in xy]
        if manual:
            dx = -dx
            dy = -h - dy
        dw = abs(dx)
        dh = abs(dy)

        # Get upper part of bottom image
        x1 = dx if dx >= 0 else 0
        y1 = 0
        x2 = w + (dx if dx < 0 else 0)
        y2 = h - dh
        box1 = (x1, y1, x2, y2)

        # Get lower part of top image
        x1 = 0
        y1 = dh
        x2 = w - dw
        y2 = h
        box2 = (x1, y1, x2, y2)

        #im_lower.crop(box1).show()
        #im_upper.crop(box2).show()
        #input("paused")

        self._validate_boxes(box1, box2)

        neighbor = tile[0], tile[1] - 1
        return ((tile, box1), (neighbor, box2))


    def _col_compare(self, xy, tile, fp_right, fp_left, manual=False):
        """Compares overlapping sections of tiles in adjacent columns

        Args:
            xy (list): offsets as (dx, dy)
            tile (list): (row, col) of primary image
            fp_right: path to primary (right) image
            fp_left: path to neighbor to the left
        """

        # Load images
        im_right = read_image(fp_right)
        im_left = read_image(fp_left)
        if im_right.size != im_left.size:
            raise ValueError("images are different sizes")
        w, h = im_right.size

        # Offsets set manually differ from those calculated by OpenCV.
        # Standardize the offsets here to the OpenCV way.
        dx, dy = [int(n) for n in xy]
        if manual:
            dx = -w - dx
            dy = -dy
        dw = abs(dx)
        dh = abs(dy)

        # Get left part of right image
        x1 = 0
        y1 = 0
        x2 = w - dw
        y2 = h - dh
        box1 = (x1, y1, x2, y2)

        # Get right part of left image
        x1 = dw
        y1 = dh
        x2 = w
        y2 = h
        box2 = (x1, y1, x2, y2)

        #im_right.crop(box1).show()
        #im_left.crop(box2).show()
        #input("paused")

        self._validate_boxes(box1, box2)

        neighbor = tile[0] - 1, tile[1]
        return ((tile, box1), (neighbor, box2))


    @staticmethod
    def _validate_boxes(box1, box2):
        """Checks if two boxes are valid and have the same dimensions"""
        try:
            assert all([c >= 0 for c in box1]), f"negative coordinate"
            assert all([c >= 0 for c in box2]), f"negative coordinate"
            assert box1[2] - box1[0] == box2[2] - box2[0], "mismatched width"
            assert box1[3] - box1[1] == box2[3] - box2[1], "mismatched height"
        except AssertionError:
            print(box1)
            print(box2)
            raise


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
                #img = cv2.imread(fp, 0)
                img = np.array(read_image(fp, mode='I'), dtype=np.uint8)
                if kwargs['scalar'] < 1:
                    h, w = [int(n*kwargs['scalar']) for n in img.shape]
                    if h and w:
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
            # FLANN-based matching. Segfaults in Ubuntu 14.04, does not
            # work with OpenCV 3.1.0 :(.
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

        # Identify inliers using homography
        if kwargs['homography'] and len(good) >= 5:
            src_pts = np.float32([kp1[m.queryIdx].pt
                                  for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt
                                  for m in good]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            try:
                matchesMask = mask.ravel().tolist()
            except AttributeError:
                return (0, 0), 0, len(matches)

            # Use inlier list to determine offset
            i = 0
            while i < len(matchesMask):
                c1 = good[i].queryIdx
                c2 = good[i].trainIdx
                x.append((kp1[c1].pt[0] - kp2[c2].pt[0]) // kwargs['scalar'])
                y.append((kp1[c1].pt[1] - kp2[c2].pt[1]) // kwargs['scalar'])
                i += 1
            x_avg = sum(x) // len(x)
            y_avg = sum(y) // len(y)

            # Return coordinates, total size, and cluster size
            n = len(x)
            m = n #len(matches)
            dt = datetime.now() - start_time
            cprint(('Matched {} features in {} and {}'
                    ' (t={})').format(n, fn1, fn2, dt), self.normal)
            return (x_avg, y_avg), n, m
        else:
            # Identify inliers using a simple clustering approach
            for m in good:
                c1 = m.queryIdx
                c2 = m.trainIdx
                x.append((kp1[c1].pt[0] - kp2[c2].pt[0]) // kwargs['scalar'])
                y.append((kp1[c1].pt[1] - kp2[c2].pt[1]) // kwargs['scalar'])
            if len(x) and len(y):
                groups = cluster(x, 2)
                x_max = max([len(group) for group in groups])
                group = [group for group in groups if len(group)==x_max][0]
                x_avg = sum(group) // len(group)

                groups = cluster(y, 2)
                y_max = max([len(group) for group in groups])
                group = [group for group in groups if len(group)==y_max][0]
                y_avg = sum(group) // len(group)

                # Return coordinates, total size, and cluster size
                n = len(x)
                m = min([x_max, y_max])
                dt = datetime.now() - start_time
                cprint(('Matched {}/{} features in {} and {}'
                        ' (t={})').format(m, n, fn1, fn2, dt), self.normal)
                return (x_avg, y_avg), n, m


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
        try:
            pct_cluster = n_cluster / float(n_total)
        except ZeroDivisionError:
            pct_cluster = 0
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
        x_min = min([coordinate[0] for coordinate in list(coordinates.values())])
        x_max = max([coordinate[0] for coordinate in list(coordinates.values())])
        y_min = min([coordinate[1] for coordinate in list(coordinates.values())])
        y_max = max([coordinate[1] for coordinate in list(coordinates.values())])

        # Shift everything so that minimum coordinates are 0,0
        for tile in coordinates:
            x, y = coordinates[tile]
            coordinates[tile] = int(x - x_min), int(y - y_min)

        return {'coordinates': coordinates, 'overlaps': self.overlaps}


def mosey(path=None, output='.', param_file='params.json', skip_file=None,
          create_jpeg=False, opencv=True, label=None, **kwargs):
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
        label (str): name of the mosaic (typically the sample name)
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
        path = _select_folder()
    # Check for tiles. If none found, try the parent directory.
    try:
        tilesets = [os.path.join(path, dn) for dn in os.listdir(path)
                    if os.path.isdir(os.path.join(path, dn))
                    and not dn == 'skipped']
    except TypeError:
        raise Exception('No filepath provided! Exiting')
    if not tilesets:
        cprint('No subdirectories found in {}. Processing'
               ' main directory instead.'.format(path))
        tilesets = [path]
    # Move tilesets most likely to yield good mosaics to front
    keys = [
        'Al',
        'Fe',
        'Ca',
        'Mg',
        'ppl',
        'refgrey',
        'nbsed',
        'grey',
        'bsed',
    ]
    tilesets = sort_tilesets(tilesets, keys=keys)
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
    positions = None
    param_file = os.path.join(output, param_file)
    for path in tilesets:
        cprint('New tileset: {} ({})'.format(os.path.basename(path), path))
        # Check for element in foldername
        mosaic = Mosaic(path, output=output, param_file=param_file,
                        skip_file=skip_file, label=label, **kwargs)
        if mosaic.grid:
            positions = mosaic.prepare_mosaic(param_file, opencv, **kwargs)
            if not positions:
                positions = mosaic.prepare_mosaic(param_file, opencv, **kwargs)
            mosaic.create_mosaic(positions, create_jpeg=create_jpeg)
        else:
            print(u'Skipped {}'.format(path))
    # Remove parameters file
    try:
        os.remove(param_file)
    except OSError:
        pass



def fingerprint(tilesets):
    """Charcterizes a list of tilesets"""
    fingerprints = {}
    exts = []
    for path in tilesets:
        exts.append(_guess_extension(path))
    for path in tilesets:
        print("Characterizing {}...".format(path))
        tiles = []
        for ext in set(exts):
            tiles.extend(list(glob.iglob(os.path.join(path, '*' + ext))))

        # Check size of first, middle, and last tiles
        sizes = []
        for tile in (tiles[0], tiles[len(tiles) // 2], tiles[-1]):
            sizes.append(Image.open(tile).size)

        fingerprints[path] = (len(tiles), list(set(sizes)))
    return fingerprints


def sort_tilesets(tilesets, keys=None):
    """Sorts a list of tilesets"""
    if keys is None:
        keys = []
    groups = {}
    for path, key in fingerprint(tilesets).items():
        groups.setdefault(serialize.dumps(key), []).append(path)
    tilesets = []
    for group in groups.values():
        for key in keys:
            for path in group[:]:
                if re.search(r'(\b|_){}(\b|_)'.format(key), path, flags=re.I):
                    group.insert(0, group.pop(group.index(path)))
        tilesets.extend(group)
    #for i, tileset in enumerate(tilesets):
    #    fprint = list(fingerprint([tileset]).values())[0]
    #    print('{}. {} ({})'.format(i + 1, tileset, fprint))
    return tilesets
