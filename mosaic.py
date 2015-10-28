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
except:
    print 'mosaic.py: Could not find module cv2'
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..helpers import cprint, prompt
from .offset import OffsetEngine




class Counter(dict):

    def add(self, key, val=1):
        try:
            self[key] += val
        except KeyError:
            self[key] = val




class Mosaic(object):


    def __init__(self, path, skipped, jpeg, opencv, **kwargs):
        self.jpeg = jpeg
        self.skipped = skipped
        self.opencv = opencv

        self.normal = True
        self.verbose = False

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
        # Check for job parameters
        try:
            params = pickle.load(open('params.p', 'rb'))
        except IOError:
            if self.opencv:
                cprint('Using OpenCV to stitch mosaic')
                defaults = {
                    'equalize_histogram' : False,
                    'matcher' : 'brute-force',
                    'scalar' : 0.5,
                    'threshold' : 0.7,
                }
                self.cv_params = {}
                for key in defaults:
                    try:
                        self.cv_params[key] = kwargs[key]
                    except KeyError:
                        self.cv_params[key] = defaults[key]
            self.get_tile_parameters(path)
            self.set_mosaic_parameters()
        else:
            cprint('Found parameters file')
            self.coordinates = params['coordinates']
            self.mag = params['mag']
            self.num_cols = params['num_cols']
            self.snake = params['snake']
            self.get_tile_parameters(path)





    def get_tile_parameters(self, path):
        """Determine parameters for tiles in path"""
        self.path = path
        # Test images
        files = glob.glob(os.path.join(path, '*.tif'))
        fp = files[0]
        try:
            img = Image.open(fp)
        except IOError:
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
                    cprint('Encountered unreadable tiles but could'
                           ' not fix them. Try installing ImageMagick'
                           ' and re-running this script.')
                    sys.exit()
        self.ext = os.path.splitext(fp)[1]
        self.w, self.h = img.size

        tiles = glob.glob(os.path.join(path, '*' + self.ext))
        try:
            tiles = [fp.encode('latin1').decode('latin1') for fp in tiles]
        except:
            pass
        self.tiles = self.handle_tiles(tiles)

        # Mandolin tileset if using saved parameters
        try:
            self.snake
        except AttributeError:
            pass
        else:
            self.rows = mandolin(self.tiles, self.num_cols)
            self.num_rows = len(self.rows)
            if self.snake:
                self.rows = [self.rows[i] if not i % 2
                             else self.rows[i][::-1]
                             for i in xrange(0, len(self.rows))]

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
        cprint('Set mosaic parameters:')
        try:
            self.num_cols
        except AttributeError:
            self.num_cols = prompt(' Number of columns:', '\d+')
            self.num_cols = int(self.num_cols)
        else:
            cprint((' Number of columns: {} (determined from'
                    ' filenames)').format(self.num_cols))
        self.rows = mandolin(self.tiles, self.num_cols)
        self.num_rows = len(self.rows)
        self.mag = prompt(' Magnification:', '\d+')
        #self.mag = 200
        self.mag = float(self.mag)
        self.snake = prompt(' Snake pattern?', yes_no)
        #self.snake = False
        if self.snake:
            self.rows = [self.rows[i] if not i % 2 else self.rows[i][::-1]
                         for i in xrange(0, len(self.rows))]
        # Review parameters
        cprint('Review parameters for your mosaic:')
        cprint(' Create JPEG:    {}'.format(self.jpeg))
        cprint(' Dimensions:     {}x{}'.format(self.num_cols, self.num_rows))
        cprint(' Magnification:  {}'.format(self.mag))
        cprint(' Snake :         {}'.format(self.snake))
        cprint(' Autostitch:     {}'.format(self.opencv))
        if self.opencv:
            cprint('  Equalize hist: {}'.format(self.cv_params['equalize_histogram']))
            cprint('  Matcher:       {}'.format(self.cv_params['matcher']))
            cprint('  Scalar:        {}'.format(self.cv_params['scalar']))
            cprint('  Threshold:     {}'.format(self.cv_params['threshold']))
        if prompt('Do these parameters look good?', yes_no):
            # Determine offsets
            if self.opencv:
                cprint('Determining offset...')
                self.coordinates = self.cv_coordinates()
            else:
                cprint('Setting offset...')
                self.coordinates = self.set_coordinates()
            # Record job parameters to file
            params = [
                self.filename,
                '-' * len(self.filename),
                'Dimensions: {}x{}'.format(self.num_cols, self.num_rows),
                'Magnification: {}'.format(self.mag),
                'Snake: {}'.format(self.snake),
                ''
            ]
            if self.opencv:
                params.extend([
                    'Autostitch: {}'.format(self.opencv),
                    'Equalize histogram: {}'.format(self.cv_params['equalize_histogram']),
                    'Matcher: {}'.format(self.cv_params['matcher']),
                    'Scalar: {}'.format(self.cv_params['scalar']),
                    'Threshold: {}'.format(self.cv_params['threshold']),
                    ''
                    ])
            params.append('Tile coordinates:')
            for key in sorted(self.coordinates.keys()):
                params.append('{}: {}'.format(key, self.coordinates[key]))
            #fp = os.path.join(self.path, os.pardir, self.filename + '.txt')
            fp = self.filename + '.txt'
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
                    position = 'x'.join([str(n) for n in [n_col, n_row]])
                    coordinates[position] = (x, y)
                n_col += 1
            n_row += 1
        return self.clean_coordinates(coordinates)





    def cv_coordinates(self):
        """Use SIFT to match features"""
        cprint('OpenCV: cv2.xfeatures2d.SIFT_create()', self.verbose)
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.keypoints = {}
        tiles = {}
        n_row = 0
        while n_row < len(self.rows):
            row = self.rows[n_row]
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
                    offset = self.cv_match(tile, neighbor)
                    if offset:
                        xy, n_total, n_cluster = offset
                        nxy = (xy[0]*-1, xy[1]*-1)
                        score = self.cv_reliability(n_cluster, n_total)
                        tiles[tile]['offsets'].append(['left', xy, score])
                        tiles[neighbor]['offsets'].append(['right', nxy,
                                                           score])
                if n_row:
                    neighbor = self.rows[n_row-1][n_col]
                    offset = self.cv_match(tile, neighbor)
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
                tile = tiles[self.rows[n_row][n_col]]
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
                        self.rows[y][x]
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
        coordinates = self.clean_coordinates(coordinates)
        return coordinates




    def analyze_offsets(self, tiles):
        for tile in self.tiles:
            tile = tiles[tile]
            n_row, n_col = tile['position']
            offsets = tile['offsets']




    def cv_match(self, img1, img2):
        start_time = datetime.now()
        # Read OpenCV params from class
        eqhist = self.cv_params['equalize_histogram']
        matcher = self.cv_params['matcher']
        scalar = self.cv_params['scalar']
        threshold = self.cv_params['threshold']

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
                if scalar < 1:
                    w = int(self.w*scalar)
                    h = int(self.h*scalar)
                    cprint('OpenCV: cv2.resize()', self.verbose)
                    img = cv2.resize(img, (w, h))
                if eqhist:
                    cprint('OpenCV: cv2.equalizeHist()', self.verbose)
                    img = cv2.equalizeHist(img)
                cprint('OpenCV: cv2.detectAndCompute()', self.verbose)
                self.keypoints[fp] = self.detector.detectAndCompute(img, None)
                del img

        kp1, des1 = self.keypoints[img1]
        if not (any(kp1) and np.any(des1)):
            cprint('No keypoints found in {}'.format(fn1), self.normal)
            return None

        kp2, des2 = self.keypoints[img2]
        if not (any(kp2) and np.any(des2)):
            cprint('No keypoints found in {}'.format(fn2), self.normal)
            return None

        if matcher == 'flann':
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
            if m.distance < threshold * n.distance:
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
            x.append((kp1[c1].pt[0] - kp2[c2].pt[0]) / scalar)
            y.append((kp1[c1].pt[1] - kp2[c2].pt[1]) / scalar)
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




    def cv_reliability(self, n_cluster, n_total):
        """Assess reliability of offset"""
        # Offset reliability test
        #  1. Are there a large number of matches?
        #  2. Do most of these matches cluster in one group?
        #  Minima: 4/5, 4/6, 5/7, 5/8, 6/9, then >50% for 10+
        if n_total < 5:
            return 0
        else:
            pct_cluster = n_cluster / float(n_total)
            return n_cluster * pct_cluster




    def create_mosaic(self, path=None, label=True):
        """Create a mosaic from a set a tiles"""
        # Normalize coordinates and calculate dimensions. The
        # dimensions of the mosaic are determined by the tile
        # dimensions minus the offsets between rows and columns
        # Some general notes:
        #  * Coordinates increase from (0,0) in the top left corner
        #  * Offsets are always applied as n - 1 because they occur
        #    between tiles.
        start_time = datetime.now()

        mosaic_width = max([coordinate[0] for coordinate in
                            self.coordinates.values()]) + self.w
        mosaic_height = max([coordinate[1] for coordinate in
                             self.coordinates.values()]) + self.h
        if label:
            label_height = int(mosaic_height * 0.04)
            mosaic_height += label_height
        cprint('Mosaic will be {:,} by {:,}'
               ' pixels'.format(mosaic_width, mosaic_height))
        # Create the mosaic
        cprint('Stitching mosaic...')
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255,255,255))
        n_row = 0
        while n_row < len(self.rows):
            row = self.rows[n_row]
            n_col = 0
            while n_col < len(row):
                position = 'x'.join([str(n) for n in [n_col, n_row]])
                try:
                    x, y = self.coordinates[position]
                except KeyError:
                    n_col += 1
                    continue
                else:
                    fp = self.rows[n_row][n_col]
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
            try:
                text = self.name
            except AttributeError:
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
        cprint('Saving as {}...'.format(self.extmap[self.ext]))
        #fp = os.path.join(self.path, os.pardir, self.filename + '.tif')
        fp = self.filename + '.tif'
        mosaic.save(fp, self.extmap[self.ext])
        if self.jpeg and self.extmap[self.ext] != 'JPEG':
            cprint('Saving as JPEG...')
            fp = os.path.splitext(fp)[0] + '.jpg'
            try:
                mosaic = mosaic.resize((mosaic_width / 2, mosaic_height / 2))
            except:
                pass
            mosaic.save(fp, 'JPEG')
        cprint('Mosaic complete! (t={})'.format(datetime.now() - start_time))
        try:
            shutil.rmtree(os.path.join(self.path, 'working'))
        except OSError:
            pass
        return self




    def handle_tiles(self, tiles, test_coherence=True):
        """Sorts and tests coherence of tileset

        @param list
        @return list

        The sort function works by detecting the iterator, which
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
                    e = ('Warning: Could not sort tiles. You'
                         ' may want to check that there are no'
                         ' extra tiles in the source folder.')
            temp[i] = tile
        cprint(e)
        if len(cols):
            try:
                self.num_cols
            except AttributeError:
                self.num_cols = max(cols) + 1
            for key in temp.keys():
                x, y = key.split(' ')
                i = int(x) + self.num_cols * int(y)
                temp[i] = temp[key]
                del temp[key]
        # Create tiles and rows
        tiles = [temp[key] for key in sorted(temp.keys())]
        # Check the tileset for coherence if required
        if test_coherence:
            tiles = self.patch_tiles(tiles)
            keys = temp.keys()
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
        return tiles




    def patch_tiles(self, tiles):
        """Returns sorted list of tiles including patches"""
        try:
            self.skipped
        except AttributeError:
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
                except KeyError:
                    sequence[i] = tile
                    break
                else:
                    i += 1
        tiles = []
        for i in sorted(sequence.keys()):
            tiles.append(sequence[i])
        cprint('Tile set was patched!')
        return tiles




    def clean_coordinates(self, coordinates):
        """Shift coordinates so minimum on each axis is 0"""
        x_min = min([coordinate[0] for coordinate in coordinates.values()])
        x_max = max([coordinate[0] for coordinate in coordinates.values()])
        y_min = min([coordinate[1] for coordinate in coordinates.values()])
        y_max = max([coordinate[1] for coordinate in coordinates.values()])

        # Shift everything so that minimum coordinates are 0,0
        for tile in coordinates:
            x, y = coordinates[tile]
            coordinates[tile] = int(x - x_min), int(y - y_min)

        return coordinates




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





def mosey(path=None, skipped=None, jpeg=False, opencv=False, **kwargs):
    """Helper function for stitching a set of directories all at once"""
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
        if 'bsed' in path or 'ppl' in path:
            tilesets.insert(0, tilesets.pop(tilesets.index(path)))
    if len(skiplists) > 1:
        cprint('Warning: Multiple skip lists found:\n ' + ' \n'.join(skiplists))
    if skipped:
        cprint('Using list of skipped tiles'
               ' from {}'.format(skiplists.pop(0)))
    for path in tilesets:
        cprint('New tileset: {}'.format(os.path.basename(path)))
        Mosaic(path, skipped, jpeg, opencv, **kwargs).create_mosaic()
    # Remove parameters file
    try:
        os.remove('params.p')
    except OSError:
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
    cprint('There was a problem opening some of the tiles!\n'
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
