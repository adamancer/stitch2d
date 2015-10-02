#Mosaic 0.92
#Updated 21 Jul 2014
#Stitches mosaics from tiles collected using SEM or Prior stage. Processes all
#images found in subfolders in PATH_ROOT in config file. Each tileset should
#be in its own folder named as follows: [prefix][number]_[suffix][type].
#Types should EXACTLY MATCH keys from the scope_type dictionary.

import csv
import glob
import os
#import psutil
import re
import subprocess
import shlex
import shutil
import sys
import tempfile
import time
import urllib
from datetime import datetime
from math import floor, sqrt
import offset
from PIL import Image, ImageDraw, ImageFont

import Tkinter
import tkFileDialog
from copy import copy

from natsort import natsorted

from ..helpers import prompt


#-------------------------------------------------------------------------------
#-FUNCTION DEFINITIONS----------------------------------------------------------
#-------------------------------------------------------------------------------


class Counter(dict):

    def add(self, key, val=1):
        try:
            self[key] += val
        except:
            self[key] = val




class Mosaic(object):


    def __init__(self, path):
        # A quick list of properties of the Mosaic object:
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
        self.extmap = {
            '.jpg' : 'JPEG',
            '.tif' : 'TIFF',
            '.tiff' : 'TIFF'
        }
        self.basepath = os.path.dirname(__file__)
        self.image_types = {}
        with open(os.path.join(self.basepath, 'files', 'image_types.txt')) as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            self.image_types = dict([(row[0].lower(), row[1])
                                      for row in rows if bool(row[0])
                                      and not row[0].startswith('#')])
        self.get_tile_parameters(path)
        self.set_mosaic_parameters()




    def get_tile_parameters(self, path):
        """Determine parameters for tiles in path"""
        exts = Counter()
        dims = Counter()
        tiles = []
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            try:
                img = Image.open(fp)
            except:
                continue
            else:
                exts.add(os.path.splitext(fn)[1])
                dims.add('x'.join([str(x) for x in img.size]))
                tiles.append(fp)
        self.ext = [key for key in exts
                    if exts[key] == max(exts.values())][0].lower()
        dim = [key for key in dims if dims[key] == max(dims.values())][0]
        self.w, self.h = [int(x) for x in dim.split('x')]
        self.filename = os.path.basename(path)
        try:
            self.name, image_type = self.filename.rsplit('_', 1)
            self.name += ' ({})'.format(self.image_types[image_type.lower()])
        except ValueError:
            # No suffix found
            pass
        except KeyError:
            # Suffix was not recognized
            pass
        self.tiles = self.sort_tiles([tile for tile in tiles
                                      if tile.endswith(self.ext)])
        return self




    def set_mosaic_parameters(self):
        """Prompt user for job parameters"""
        yes_no = {'y' : True, 'n' : False}
        try:
            print ('Number of columns detected from filenames'
                   ' (n={})').format(self.num_cols)
        except AttributeError:
            self.num_cols = int(prompt('Number of columns:', '\d+'))
        self.mag = int(prompt('Magnification:', '\d+'))
        self.snake = prompt('Snake pattern?', yes_no)
        self.rows = self.mandolin(self.tiles, self.num_cols)
        self.num_rows = len(self.rows)
        self.determine_offset()
        if prompt('Are these parameters okay?', yes_no):
            return self
        else:
            del self.num_cols
            self.set_mosaic_parameters()




    def create_mosaic(self, path=None, label=True):
        """Create a mosaic from a set a tiles with known, fixed offsets"""
        start_time = datetime.now()
        # Folder-specific parameters are cleared when the mosaic is
        # done stitching, but most parameters carry over, allowing a
        # bunch of mosaics to be run at once with the same settings.
        if path:
            self.get_tile_parameters(path)
            self.rows = self.mandolin(self.tiles, self.num_cols)
            self.num_rows = len(self.rows)
        # The dimensions of the mosaic are determined by the
        # tile dimensions minus the offsets between rows and
        # columns. Some general notes:
        #   * The origin is in the upper left
        #   * Offsets are always applied as n - 1 because they occur
        #     between tiles
        mosaic_width = (self.w * self.num_cols +
                        self.x_offset_within_row * (self.num_cols - 1) +
                        abs(self.x_offset_between_rows) * (self.num_rows - 1))
        mosaic_height = (self.h * self.num_rows +
                         self.y_offset_between_rows * (self.num_rows - 1) +
                         abs(self.y_offset_within_row) * (self.num_cols - 1))
        if label:
            label_height = int(mosaic_height * 0.04)
            mosaic_height += label_height
        print 'Mosaic will be {:,} by {:,} pixels'.format(mosaic_width,
                                                          mosaic_height)
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), (255,255,255))
        # Now that the canvas has been created, we can paste the
        # individual tiles on top of it. Coordinates increase from
        # (0,0) in the top left corner.
        print 'Stitching mosaic...'
        n_row = 0  # index of row
        for row in self.rows:
            if self.snake and not (n_row + 1) % 2:
                row = row[::-1]
            n_col = 0  # index of column
            for fp in row:
                if bool(fp):
                    # Calculate x coordinate
                    x = ((self.w + self.x_offset_within_row) * n_col +
                         self.x_offset_between_rows * n_row)
                    if self.x_offset_between_rows < 0:
                        x -= self.x_offset_between_rows * (self.num_rows - 1)
                    # Calculate y coordinate
                    y = ((self.h + self.y_offset_between_rows) * n_row +
                         self.y_offset_within_row * n_col)
                    if self.y_offset_within_row < 0:
                        y -= self.y_offset_within_row * (self.num_cols - 1)
                    mosaic.paste(Image.open(fp), (x, y))
                n_col += 1
            n_row += 1
        # Add label
        if label:
            try:
                text = self.name
            except:
                text = self.filename
            text = text[0].upper() + text[1:]
            draw = ImageDraw.Draw(mosaic)
            # Resize text to a reasonable size based on the
            # dimensions of the mosaic
            size = 100
            font = ImageFont.truetype('Microsoft Sans Serif.ttf', size)
            w, h = font.getsize(text)
            size = int(0.8 * size * label_height / float(h))
            font = ImageFont.truetype('Microsoft Sans Serif.ttf', size)
            x = int(0.02 * mosaic_width)
            y = mosaic_height - int(label_height)
            draw.text((x, y), text, (0, 0, 0), font=font)
        mosaic.save(self.filename + '.tif', self.extmap[self.ext])
        print 'Stitching complete! (t={})'.format(datetime.now() - start_time)
        # Clear folder-specific parameters. These will be repopulated
        # automatically if the same Mosaic object is used for a
        # different filepath. This isn't crucial--these values should
        # be overwritten by future uses of the same Mosaic instance.
        try:
            del self.filename
            del self.tiles
            del self.rows
            del self.name
        except AttributeError:
            pass
        return self




    def sort_tiles(self, tiles):
        """Identify iterator in filename and sort

        @param list
        @return list

        The iterator is the part of the filename that changes
        between files in the same set of tiles. Typically the
        interator will be an integer (abc-1.jpg or abc-001.jpg)
        or, using the SEM, a column-row pair (abc_Grid[@0 0].jpg).
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
        for tile in tiles:
            key = tile.replace(starts, '', 1).replace(ends, '', 1)
            try:
                x, y = key.split(' ')
                i = int(y) * self.num_cols + int(x)
                cols.append(y)
            except ValueError:
                i = key
            temp[i] = tile
        # Bonus: Determine the number of columns if the tiles are
        # provided with SEM-style grid notation. This is kind of an
        # odd fit here, but I don't know where else to put it.
        try:
            self.num_cols = max(cols) + 1
        except (UnboundLocalError, ValueError):
            pass
        return [temp[key] for key in sorted(temp.keys())]





    def determine_offset(self, same_row=True):
        """Use pyglet to allow users to set offset between tiles"""
        # Coordinates increase from 0,0 in the upper left. Offsets
        # are defined as follows:
        #  Within row: y is positive if the top edge of the right
        #   tile is HIGHER than that of the left (stair step up).
        #   Because tiles must be shifted left to overlap, x
        #   is always negative.
        #  Between rows: x is positive if the left edge of the lower
        #   tile is to the RIGHT of the left edge of the upper tile.
        #   Because tiles must be shifted up to overlap, y is
        #   always negative.
        self.x_offset_within_row = -100  # must be negative
        self.x_offset_between_rows = -50
        self.y_offset_within_row = -25
        self.y_offset_between_rows = -124  # must be positive
        return self




    def mandolin(self, lst, n):
        """Split list into groups of n members

        @param list
        @param int
        @return list
        """
        return [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]




    def patch_tiles(self, tiles):
        """Patch tileset with blanks based on mosaic.Selector"""
        """Returns sorted list of tiles including patches"""
        patches = glob.glob(os.path.join(path, '..', 'output',
                                         'placeholders', '*' + ext))
        if len(patches):
            print 'Patching mosaic with tiles set aside by selector.py...'
        # Insert patches into sequence
        sequence = {}
        for tile in patches:
            coords = os.path.splitext(tile)[0].split('@')[1].rstrip(']').split(' ')
            y, x = [int(x) for x in coords]
            i = x * num_cols + y
            sequence[i] = tile
        tiles = glob.glob(os.path.join(path, '*' + ext))
        tiles.sort(key=lambda fn: sorter(fn))
        j = 0
        for tile in tiles:
            while True:
                try:
                    sequence[j]
                except:
                    sequence[j] = tile
                    break
                else:
                    j += 1
        tiles = []
        for i in sorted(sequence.keys()):
            tiles.append(sequence[i])
        #print '\n'.join(tiles)
        return tiles
