import csv
import glob
import os
import shutil
import sys
import time
import Tkinter
import tkFileDialog
from copy import copy
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from textwrap import fill

from ..helpers import prompt
from .offset import OffsetEngine




class Counter(dict):

    def add(self, key, val=1):
        try:
            self[key] += val
        except:
            self[key] = val




class Mosaic(object):


    def __init__(self, path, jpeg=False, skipped=None):
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
        with open(os.path.join(self.basepath, 'files', 'image_types.txt')) as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            self.image_types = dict([(row[0].lower(), row[1])
                                      for row in rows if bool(row[0])
                                      and not row[0].startswith('#')])
        self.skipped = skipped
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
        print 'Set mosaic parameters:'
        try:
            self.num_cols
        except:
            self.num_cols = prompt(' Number of columns:', '\d+')
            self.num_cols = int(self.num_cols)
        else:
            print (' Number of columns: {} (determined from'
                   ' filenames)').format(self.num_cols)
        self.mag = prompt(' Magnification:', '\d+')
        self.mag = int(self.mag)
        self.snake = prompt(' Snake pattern?', yes_no)
        self.rows = mandolin(self.tiles, self.num_cols)
        self.num_rows = len(self.rows)
        print 'Setting offset...'
        engine = OffsetEngine(self.rows, True)
        offset = engine.set_offset()
        engine = OffsetEngine(self.rows, False, offset)
        offset = engine.set_offset()
        self.x_offset_within_row = offset[0]
        self.y_offset_within_row = offset[1]
        self.x_offset_between_rows = offset[2]
        self.y_offset_between_rows = offset[3]
        # Review parameters
        print 'Review parameters for your mosaic:'
        print ' Dimensions:   ', '{}x{}'.format(self.num_cols, self.num_rows)
        print ' Magnification:', self.mag
        print ' Offset:       ', offset
        print ' Snake:        ', self.snake
        print ' Create JPEG:  ', self.jpeg
        if prompt('Do these parameters look good?', yes_no):
            return self
        else:
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
                print ' Index: {}'.format(key)
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
        if not len(self.skipped):
            return tiles
        else:
            # Get grid dimensions. These can get screwed up if an
            # entire column is removed.
            self.num_cols = int(self.skipped.pop(0)
                                .split(': ', 1)[1]
                                .split('x', 1)[0])
        sequence = {}
        for i in self.skipped:
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





def mosey(path=None, jpeg=False, skipped=None):
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
    for path in copy(tilesets):
        if not skipped:
            try:
                skipped = handle_skipped(path)
            except IOError:
                pass
        if 'bsed' in path or '_Si' in path:
            tilesets.insert(0, tilesets.pop(tilesets.index(path)))
    for path in tilesets:
        print '-' * 80
        print 'New tileset: {}'.format(os.path.basename(path))
        try:
            mosaic.create_mosaic(path)
        except NameError:
            mosaic = Mosaic(path, jpeg, skipped).create_mosaic()




def handle_skipped(path):
    try:
        f = open(os.path.join(path, 'skipped.txt'), 'rb')
    except:
        return []
    else:
        try:
            return [int(i.strip()) if i.isdigit() else i.strip()
                    for i in f.read().splitlines()]
        except TypeError:
            raise




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
