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
        self.path = path
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
                tiles.append(fp.encode('latin1').decode('latin1'))
        self.ext = [key for key in exts
                    if exts[key] == max(exts.values())][0].lower()
        dim = [key for key in dims if dims[key] == max(dims.values())][0]
        self.w, self.h = [int(x) for x in dim.split('x')]
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
        tiles = [tile for tile in tiles if tile.endswith(self.ext)]
        self.handle_tiles(tiles)
        return self




    def set_mosaic_parameters(self):
        """Prompt user for job parameters"""
        yes_no = {'y' : True, 'n' : False}
        try:
            self.num_cols
        except:
            self.num_cols = int(prompt('Number of columns:', '\d+'))
        else:
            print ('Columns determined from'
                   ' filenames (n={})').format(self.num_cols)
        self.mag = int(prompt('Magnification:', '\d+'))
        self.snake = prompt('Snake pattern?', yes_no)
        self.rows = mandolin(self.tiles, self.num_cols)
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
            self.rows = mandolin(self.tiles, self.num_cols)
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
                    # Encode the name
                    mosaic.paste(Image.open(fp.encode('cp1252')), (x, y))
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
            font = ImageFont.truetype('arial.ttf', size)
            w, h = font.getsize(text)
            size = int(0.8 * size * label_height / float(h))
            font = ImageFont.truetype('arial.ttf', size)
            x = int(0.02 * mosaic_width)
            y = mosaic_height - int(label_height)
            draw.text((x, y), text, (0, 0, 0), font=font)
        fp = os.path.join(self.path, os.pardir, self.filename + '.tif')
        mosaic.save(fp, self.extmap[self.ext])
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
                i = key
            temp[i] = tile
        if len(cols):
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
                print ' Index: {}'.format(key)
        else:
            print 'All tiles appear to be present'
        self.tiles = tiles





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
        self.x_offset_within_row = -190  # must <= 0
        self.x_offset_between_rows = -6
        self.y_offset_within_row = -4
        self.y_offset_between_rows = -56  # must <= 0
        return self




    def patch_tiles(self, tiles):
        """Returns sorted list of tiles including patches"""
        try:
            f = open(os.path.join(self.path, 'skipped.txt'), 'rb')
        except:
            return tiles
        sequence = {}
        skipped = [int(i.strip()) for i in f.read().splitlines()]
        for i in skipped:
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




def mosey(path=None):
    """Helper function for stitching a set of directories all at once"""
    if not path:
        root = Tkinter.Tk()
        root.withdraw()
        title = "Please select the directory containing your tile sets:"
        initial = os.path.expanduser('~')
        path = tkFileDialog.askdirectory(parent=root, title=title,
                                         initialdir=initial)
    print path
    for path in ([os.path.join(path, dn) for dn in os.listdir(path)
                  if os.path.isdir(os.path.join(path, dn))]):
        print 'New tileset: {}'.format(os.path.basename(path))
        try:
            mosaic.create_mosaic(path)
        except NameError:
            mosaic = Mosaic(path).create_mosaic()




def mandolin(lst, n):
    """Split list into groups of n members

    @param list
    @param int
    @return list
    """
    return [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]
