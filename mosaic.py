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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..helpers import prompt
from .offset import OffsetEngine




class Counter(dict):

    def add(self, key, val=1):
        try:
            self[key] += val
        except:
            self[key] = val




class Mosaic(object):


    def __init__(self, path, opencv=False, jpeg=False, skipped=None):
        # Properties of the Mosaic object:
        #  self.filename (str, specific)
        #  self.name (str, specific)
        #  self.tiles (list, specific)
        #  self.rows (int, specific)
        #  self.basepath (str, carries over)
        #  self.image_types (dict, carries over)
        #  self.extmap (dict, carries over)
        #  self.snake (bool, carries over)
        #  self.ext (str, carries over)
        #  self.w (int, carries over)
        #  self.h (int, carries over)
        #  self.mag (int, carries over)
        #  self.num_rows (int, carries over)
        #  self.num_cols (int, carries over)
        #  self.x_offset_within_row (int, carries over)
        #  self.x_offset_between_rows (int, carries over)
        #  self.y_offset_within_row (int, carries over)
        #  self.y_offset_between_rows (int, carries over)
        self.jpeg = jpeg
        self.extmap = {
            '.jpg' : 'JPEG',
            '.tif' : 'TIFF',
            '.tiff' : 'TIFF'
        }
        self.basepath = os.path.dirname(__file__)
        self.image_types = {}
        with open(os.path.join(self.basepath,
                               'files', 'image_types.txt')) as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            self.image_types = dict([(row[0].lower(), row[1])
                                      for row in rows if bool(row[0])
                                      and not row[0].startswith('#')])
        self.skipped = skipped
        self.opencv = opencv
        # Check for job parameters
        self.get_tile_parameters(path)
        try:
            f = open('params.p')
        except:
            self.set_mosaic_parameters()
        else:
            print 'Found parameters file'
            params = pickle.load(f)
            self.coordinates = params['coordinates']
            self.mag = params['mag']
            self.num_cols = params['num_cols']
            self.snake = params['snake']




    def get_tile_parameters(self, path):
        """Determine parameters for tiles in path"""
        self.path = path
        # Test images
        files = glob.glob(os.path.join(path, '*.tif'))
        fp = files[0]
        try:
            img = Image.open(fp)
        except:
            # This is a clumsy solution to PIL's unreliability
            # reading TIFFs. It uses ImageMagick to copy
            # the unreadable tiles to a subdirectory; the
            # IM-created tiles should always be readable by PIL.
            ext = os.path.splitext(fp)[1].lower()
            if ext in self.extmap:
                if mogrify(path, ext):
                    path = os.path.join(path, 'working')
                    img = Image.open(os.path.join(path, os.listdir(path)[0]))
                else:
                    print ('Encountered unreadable tiles but could'
                           ' not fix them. Try installing ImageMagick'
                           ' and re-running this script.')
        self.ext = os.path.splitext(fp)[1]
        self.w, self.h = img.size

        tiles = glob.glob(os.path.join(path, '*' + self.ext))
        tiles = [fp.encode('latin1').decode('latin1') for fp in tiles]
        self.handle_tiles(tiles)

        # Mandolin tileset if using saved parameters
        try:
            self.rows = mandolin(self.tiles, self.num_cols)
        except AttributeError:
            pass
        else:
            self.num_rows = len(self.rows)

        if path.strip('/').endswith('working'):
            path = os.path.split(path)[0]

        # Get name
        self.filename = unicode(os.path.basename(path))
        try:
            self.name, image_type = self.filename.rsplit('_', 1)
            self.name += ' ({})'.format(self.image_types[image_type.lower()])
        except ValueError:
            # No suffix found
            pass
        except KeyError:
            # Suffix was not recognized
            pass

        return self




    def set_mosaic_parameters(self):
        """Prompt user for job parameters"""
        yes_no = {'y' : True, 'n' : False}
        print 'Set mosaic parameters:'
        try:
            self.num_cols
        except:
            self.num_cols = prompt(' Number of columns:', '\d+')
            self.num_cols = int(self.num_cols)
        else:
            print (' Number of columns: {} (determined from'
                   ' filenames)').format(self.num_cols)
        self.rows = mandolin(self.tiles, self.num_cols)
        self.num_rows = len(self.rows)
        self.mag = prompt(' Magnification:', '\d+')
        self.mag = float(self.mag)
        self.snake = prompt(' Snake pattern?', yes_no)
        if self.snake:
            self.rows = [self.rows[i] if not i % 2 else self.rows[i][::-1]
                         for i in xrange(0, len(self.rows))]
        # Determine offsets
        if self.opencv:
            print 'Determining offset...'
            self.coordinates = self.cv_coordinates()
        else:
            print 'Setting offset...'
            self.coordinates = self.set_coordinates()
        # Review parameters
        print 'Review parameters for your mosaic:'
        print ' Dimensions:         ', '{}x{}'.format(self.num_cols,
                                                      self.num_rows)
        print ' Magnification:      ', self.mag
        print ' Snake:              ', self.snake
        print ' Create JPEG:        ', self.jpeg
        if prompt('Do these parameters look good?', yes_no):
            # Write job parameters to file. This could be embedded in
            # the metadata.
            params = [
                self.filename,
                '-' * len(self.filename),
                'Dimensions: {}x{}'.format(self.num_cols, self.num_rows),
                'Magnification: {}'.format(self.mag),
                'Snake: {}'.format(self.snake),
                '',
                'Tile coordinates:'
            ]
            for key in sorted(self.coordinates.keys()):
                params.append('{}: {}'.format(key, self.coordinates[key]))
            fp = os.path.join(self.path, os.pardir, self.filename + '.txt')
            with open(fp, 'wb') as f:
                f.write('\n'.join(params))
            # Pickle key parameters for re-use later
            params = {
                'coordinates' : self.coordinates,
                'mag' : self.mag,
                'num_cols' : self.num_cols,
                'snake' : self.snake
            }
            with open('params.p', 'wb') as f:
                pickle.dump(params, f)
            return self
        else:
            del self.num_cols
            self.set_mosaic_parameters()




    def set_coordinates(self):
        """Set coordinates using user-defined offsets"""
        # Use the offset engine to allow users to set the offsets
        engine = OffsetEngine(self.rows, True)
        offset = engine.set_offset()
        engine = OffsetEngine(self.rows, False, offset)
        offset = engine.set_offset()

        x_offset_within_row = offset[0]
        y_offset_within_row = offset[1]
        x_offset_between_rows = offset[2]
        y_offset_between_rows = offset[3]

        coordinates = {}
        n_row = 0  # index of row
        for row in self.rows:
            n_col = 0  # index of column
            for fp in row:
                if bool(fp):
                    # Calculate x coordinate
                    x = ((self.w + x_offset_within_row) * n_col +
                         x_offset_between_rows * n_row)
                    if x_offset_between_rows < 0:
                        x -= x_offset_between_rows * (self.num_rows - 1)
                    # Calculate y coordinate
                    y = ((self.h + y_offset_between_rows) * n_row +
                         y_offset_within_row * n_col)
                    if y_offset_within_row < 0:
                        y -= y_offset_within_row * (self.num_cols - 1)
                    coordinates[fp] = (x, y)
                n_col += 1
            n_row += 1
        return self.clean_coordinates(coordinates)




    def cv_coordinates(self):
        """Automatically determine offsets using opencv"""
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.keypoints = {}
        offsets = {}

        # Reliability test
        #  1. Are there a large number of matches?
        #  2. Are most of those matches part of the same cluster?
        #  3. Is the within-row or between-row offset more reliable?

        n_row = 0
        while n_row < len(self.rows):
            within_row = {}
            between_rows = {}
            keys = []
            row = self.rows[n_row]
            n_col = 0
            while n_col < len(row):
                tile = row[n_col]
                keys.append(tile)
                if n_col:
                    neighbor = row[n_col-1]
                    within_row[tile] = self.cv_offset(tile, neighbor)
                else:
                    within_row[tile] = (-1, (0,0))
                if n_row:
                    neighbor = self.rows[n_row-1][n_col]
                    between_rows[tile] = self.cv_offset(tile, neighbor)
                else:
                    between_rows[tile] = (-1, (0,0))
                n_col += 1

            # Find best within-row offset
            try:
                mx = max([offset[0] for offset in within_row.values()])
            except:
                offset1 = (0,0)
            else:
                offset1 = [offset[1] for offset in within_row.values()
                           if offset[0] == mx][0]
                offset1 = (-(offset1[0] + self.w), -offset1[1])

            # Find best between-row offset
            try:
                mx = max([offset[0] for offset in between_rows.values()])
            except:
                offset2 = (0,0)
            else:
                offset2 = [offset[1] for offset in between_rows.values()
                           if offset[0] == mx][0]
                offset2 = (-offset2[0], -(offset2[1] + self.h))

            # Adjust between row offset
            offset = []
            for key in keys:
                offsets[key] = (offset1, offset2)
            n_row += 1

        # Get coordinates
        coordinates = {}
        n_row = 0
        for row in self.rows:
            n_col = 0
            while n_col < len(row):
                tile = self.rows[n_row][n_col]
                wr, br = offsets[tile]
                wrx, wry = wr  # within-row offset
                brx, bry = br  # between-row offset
                if not n_row and not n_col:
                    coordinates[tile] = (0, 0)
                elif not n_col:
                    # Apply between-row offset to first tile in row
                    neighbor = coordinates[self.rows[n_row-1][n_col]]
                    x = neighbor[0] + brx
                    y = neighbor[1] + self.h + bry
                    coordinates[tile] = (x, y)
                else:
                    # Apply within-row offset to subsequent tiles in row
                    neighbor = coordinates[self.rows[n_row][n_col-1]]
                    x = neighbor[0] + self.w + wrx
                    y = neighbor[1] + wry
                    coordinates[tile] = (x, y)
                n_col += 1
            n_row += 1

        return self.clean_coordinates(coordinates)




    def cv_offset(self, img1, img2):

        # Read descriptors
        for fp in (img2, img1):
            if not bool(fp):
                return -1, (0, 0)
            try:
                self.keypoints[fp]
            except:
                print 'Getting keypoints for {}'.format(fp)
                img = cv2.imread(fp, 0)
                self.keypoints[fp] = self.sift.detectAndCompute(img, None)

        kp1, des1 = self.keypoints[img1]
        if not (any(kp1) and np.any(des1)):
            print 'No keypoints found in {}'.format(img1)
            return -1, (0,0)

        kp2, des2 = self.keypoints[img2]
        if not (any(kp2) and np.any(des2)):
            print 'No keypoints found in {}'.format(img2)
            return -1, (0,0)

        print 'Matching features in {1} and {0}'.format(img1, img2)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            print 'No matches found in {1} and {0}'.format(img1, img2)
            return -1, (0,0)

        # Matches consist of DMatch objects, which among other things
        # contain coordinates for keypoints in kp1 and kp2. These can
        # be used to calculate an average offset between two tiles;
        # the average is based on a simple cluster analysis of matches
        # detected between the two images.

        # Ratio test as per Lowe's paper
        x = []
        y = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                c1 = m.queryIdx
                c2 = m.trainIdx
                x.append(kp1[c1].pt[0] - kp2[c2].pt[0])
                y.append(kp1[c1].pt[1] - kp2[c2].pt[1])
        if len(x) and len(y):
            # Return the largest clusters for x and y
            groups = cluster(x, 2)
            x_max = max([len(group) for group in groups])
            group = [group for group in groups if len(group)==x_max][0]
            x_avg = sum(group) / len(group)

            groups = cluster(y, 2)
            y_max = max([len(group) for group in groups])
            group = [group for group in groups if len(group)==y_max][0]
            y_avg = sum(group) / len(group)

            # Log values
            with open('offsets.csv', 'ab') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([img2, x_avg, x_max, len(x)])
                writer.writerow([img2, y_avg, y_max, len(y)])

            # Return total matches, average offsets, and group sizes
            return len(x), (x_avg, y_avg), (x_max, y_max)
        return -1, (0,0)




    def clean_coordinates(self, coordinates):
        x_min = min([coordinate[0] for coordinate in coordinates.values()])
        x_max = max([coordinate[0] for coordinate in coordinates.values()])
        y_min = min([coordinate[1] for coordinate in coordinates.values()])
        y_max = max([coordinate[1] for coordinate in coordinates.values()])

        # Shift everything so that minimum coordinates are 0,0
        for tile in coordinates:
            x, y = coordinates[tile]
            coordinates[tile] = int(x - x_min), int(y - y_min)

        return coordinates




    def create_mosaic(self, path=None, label=True):
        """Create a mosaic from a set a tiles"""
        start_time = datetime.now()

        # Normalize coordinates and calculate dimensions. The
        # dimensions of the mosaic are determined by the tile
        # dimensions minus the offsets between rows and columns
        # Some general notes:
        #  * Coordinates increase from (0,0) in the top left corner
        #  * Offsets are always applied as n - 1 because they occur
        #    between tiles.
        mosaic_width = max([coordinate[0] for coordinate in
                            self.coordinates.values()]) + self.w
        mosaic_height = max([coordinate[1] for coordinate in
                             self.coordinates.values()]) + self.h
        if label:
            label_height = int(mosaic_height * 0.04)
            mosaic_height += label_height
        print 'Mosaic will be {:,} by {:,} pixels'.format(mosaic_width,
                                                          mosaic_height)
        # Create the mosaic
        print 'Stitching mosaic...'
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255,255,255))
        for fp in self.coordinates:
            x, y = self.coordinates[fp]
            # Encode the name
            try:
                mosaic.paste(Image.open(fp.encode('cp1252')), (x, y))
            except:
                print ('Encountered unreadable tiles but'
                       ' could not fix them. Try installing'
                       ' ImageMagick and re-running this'
                       ' script.')
        # Add label
        if label:
            ttf = os.path.join(self.basepath, 'files', 'OpenSans-Regular.ttf')
            try:
                text = self.name
            except:
                text = self.filename
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
        print 'Saving as {}...'.format(self.extmap[self.ext])
        fp = os.path.join(self.path, os.pardir, self.filename + '.tif')
        mosaic.save(fp, self.extmap[self.ext])
        if self.jpeg and self.extmap[self.ext] != 'JPEG':
            print 'Saving as JPEG...'
            fp = os.path.splitext(fp)[0] + '.jpg'
            try:
                mosaic = mosaic.resize((mosaic_width / 2, mosaic_height / 2))
            except:
                pass
            mosaic.save(fp, 'JPEG')
        print 'Mosaic complete! (t={})'.format(datetime.now() - start_time)
        # Clear folder-specific parameters. This isn't crucial--these
        # values should be overwritten by future uses of the same Mosaic
        # instance.
        try:
            del self.filename
            del self.tiles
            del self.rows
            del self.name
        except AttributeError:
            pass
        try:
            shutil.rmtree(os.path.join(self.path, 'working'))
        except OSError:
            pass
        return self




    def handle_tiles(self, tiles):
        """Sorts and tests coherence of tileset

        @param list
        @return list

        The sort function works by deteteching the iterator, which
        is defined here as the part of the filename that changes
        between files in the same set of tiles. Typically the
        interator will be an integer (abc-1.jpg or abc-001.jpg)
        or, using the SEM, a column-row pair (abc_Grid[@0 0].jpg).

        This function also checks the coherence of the tile set
        and warns users if tiles appear to be missing.
        """
        # First we identify this iterator by finding which parts
        # of the string are constant across the tileset.
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
                    e = ('Warning: Non-numeric iterator found.'
                         ' You may want to check that there are'
                         ' no extra tiles in the source folder.')
            temp[i] = tile
        if e:
            print fill(e, subsequent_indent=' ')
        if len(cols):
            try:
                self.num_cols
            except:
                self.num_cols = max(cols) + 1
            for key in temp.keys():
                x, y = key.split(' ')
                i = int(x) + self.num_cols * int(y)
                temp[i] = temp[key]
                del temp[key]
        # Create tiles and rows
        tiles = [temp[key] for key in sorted(temp.keys())]
        tiles = self.patch_tiles(tiles)
        # Check the tileset for coherence
        keys = temp.keys()
        idealized_keys = set([x for x in xrange(min(keys), max(keys)+1)])
        missing = idealized_keys - set(keys)
        empties = [tile for tile in tiles if not bool(tile)]
        if len(missing) and not len(empties):
            print 'Warning: The following tiles appear to be missing:'
            for key in sorted(missing):
                print 'Index {}'.format(key)
                tiles.insert(key-1, '')
        else:
            print 'All tiles appear to be present'
        self.tiles = tiles




    def patch_tiles(self, tiles):
        """Returns sorted list of tiles including patches"""
        try:
            self.skipped
        except:
            try:
                self.skipped = handle_skipped(self.path)
            except:
                return tiles
        if not self.skipped:
            return tiles
        else:
            # Get grid dimensions from skipped file, then check
            # length of tiles against the number of skipped tiles
            # to see.
            dim = self.skipped[0].split(': ', 1)[1].strip()
            self.num_cols, self.num_rows =  [int(x) for x in dim.split('x')]
            remainder = len(tiles) % self.num_cols
            n = self.num_cols * self.num_rows
            if n == len(tiles) + (self.num_cols - remainder):
                for i in self.skipped[1:]:
                    tiles[i] = ''
                return tiles
        # Insert blanks where they should fall in the tile sequence,
        # then fill the tiles around them.
        sequence = {}
        for i in self.skipped[1:]:
            sequence[i] = ''
        i = 0
        for tile in tiles:
            while True:
                try:
                    sequence[i]
                except:
                    sequence[i] = tile
                    break
                else:
                    i += 1
        tiles = []
        for i in sorted(sequence.keys()):
            tiles.append(sequence[i])
        print 'Tile set was patched!'
        return tiles




def cluster(data, maxgap):
    '''Arrange data into groups where successive elements
       differ by no more than *maxgap*
       From http://stackoverflow.com/questions/14783947
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups




def mosey(path=None, opencv=False, jpeg=False, skipped=None):
    """Helper function for stitching a set of directories all at once"""
    if not path:
        root = Tkinter.Tk()
        root.withdraw()
        title = 'Please select the directory containing your tile sets:'
        initial = os.path.expanduser('~')
        path = tkFileDialog.askdirectory(parent=root, title=title,
                                         initialdir=initial)
    # Check for tiles. If none found, try the parent directory.
    tilesets = [os.path.join(path, dn) for dn in os.listdir(path)
                if os.path.isdir(os.path.join(path, dn))
                and not dn == 'skipped']
    if not len(tilesets):
        print fill('No subdirectories found in {}. Processing main'
                   ' directory instead.'.format(path), subsequent_indent=' ')
        tilesets = [path]
    # Check for skipped files. By default, mosey will check all
    # subdirectories of the main directory for skipped file list and then
    # apply that list to everything processed in the current job. Also,
    # element maps can be hit or miss for setting offsets, so we shift
    # backscatter images to top of the list if they're available.
    skiplists = []
    for path in copy(tilesets):
        try:
            skipped = handle_skipped(path)
        except IOError:
            pass
        else:
            if skipped:
                skiplists.append(path)
        if 'bsed' in path:
            tilesets.insert(0, tilesets.pop(tilesets.index(path)))
    if len(skiplists) > 1:
        print 'Warning: Multiple skip lists found:\n ' + ' \n'.join(skiplists)
    if skipped:
        print fill('Using list of skipped tiles'
                   ' from {}'.format(skiplists.pop(0)), subsequent_indent=' ')
    for path in tilesets:
        print 'New tileset: {}'.format(os.path.basename(path))
        Mosaic(path, opencv, jpeg, skipped).create_mosaic()
    # Remove parameters file
    try:
        os.remove('params.p')
    except IOError:
        pass




def handle_skipped(path):
    try:
        f = open(os.path.join(path, 'skipped.txt'), 'rb')
    except OSError:
        raise OSError
    else:
        try:
            return [int(i.strip()) if i.isdigit() else i.strip()
                    for i in f.read().splitlines()]
        except TypeError:
            raise TypeError




def mandolin(lst, n):
    """Split list into groups of n members

    @param list
    @param int
    @return list
    """
    mandolined = [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]
    remainder = len(lst) % n
    if remainder:
        leftovers = lst[-remainder:]
        mandolined.append(leftovers + [''] * (n - len(leftovers)))
    return mandolined




def mogrify(path, ext):
    """Saves copy of tiles to subfolder"""
    print ('There was a problem opening some of the tiles!\n'
           'Copying tiles into a usable format...')
    ext = ext.strip('*.')
    subdir = os.path.join(path, 'working')
    try:
        os.mkdir(subdir)
    except OSError:
        pass
    cmd = 'mogrify -path "{0}" -format {1} *.{1}'.format(subdir, ext)
    args = shlex.split(cmd)
    try:
        subprocess.call(args, cwd=path)
    except:
        return False
    else:
        return True
