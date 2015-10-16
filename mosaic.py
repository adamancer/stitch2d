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
        self.workaround = False  # tracks TIFF reading nightmare
        self.get_tile_parameters(path)
        self.set_mosaic_parameters()




    def get_tile_parameters(self, path):
        """Determine parameters for tiles in path"""
        self.path = path
        try:
            self.ext
            self.w
            self.h
        except AttributeError:
            # Tile parameters not set, so we'll assign them now
            print 'Getting tile parameters...'
            exts = Counter()
            dims = Counter()
            tiles = []
            while True:
                for fn in os.listdir(path):
                    fp = os.path.join(path, fn)
                    try:
                        img = Image.open(fp)
                    except:
                        # This is a clumsy solution to PIL's unreliability
                        # reading TIFFs. It uses ImageMagick to copy
                        # the unreadable tiles to a subdirectory; the
                        # IM-created tiles should always be readable by PIL.
                        ext = os.path.splitext(fp)[1].lower()
                        if ext in self.extmap:
                            self.workaround = mogrify(path, ext)
                            if self.workaround:
                                path = os.path.join(path, 'working')
                                break
                            else:
                                print ('Encountered unreadable tiles but'
                                       ' could not fix them. Try installing'
                                       ' ImageMagick and re-running this'
                                       ' script.')
                                raise
                    else:
                        exts.add(os.path.splitext(fn)[1])
                        dims.add('x'.join([str(x) for x in img.size]))
                        tiles.append(fp.encode('latin1').decode('latin1'))
                else:
                    self.ext = [key for key in exts
                                if exts[key] == max(exts.values())][0].lower()
                    dim = [key for key in dims
                           if dims[key] == max(dims.values())][0]
                    self.w, self.h = [int(x) for x in dim.split('x')]
                    break
        else:
            print 'Using tile parameters from the last mosaic'
            tiles = [fp.encode('latin1').decode('latin1')
                     for fp in glob.glob(os.path.join(path, '*' + self.ext))]
            try:
                Image.open(tiles[0])
            except:
                self.workaround = mogrify(path, self.ext)
                path = os.path.join(path, 'working')
                tiles = [fp.encode('latin1').decode('latin1') for fp
                         in glob.glob(os.path.join(path, '*' + self.ext))]
        if self.workaround:
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
        self.mag = float(self.mag)
        self.snake = prompt(' Snake pattern?', yes_no)
        self.rows = mandolin(self.tiles, self.num_cols)
        if self.snake:
            for i in [i for i in xrange(0, len(self.rows)) if i % 2]:
                self.rows[i]= self.rows[i][::-1]
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
        print ' Dimensions:         ', '{}x{}'.format(self.num_cols,
                                                    self.num_rows)
        print ' Magnification:      ', self.mag
        print ' Offset within row:  ', '{}x{}'.format(
                    self.x_offset_within_row,
                    self.y_offset_within_row)
        print ' Offset between rows:', '{}x{}'.format(
                    self.x_offset_between_rows,
                    self.y_offset_between_rows)
        print ' Snake:              ', self.snake
        print ' Create JPEG:        ', self.jpeg
        if prompt('Do these parameters look good?', yes_no):
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
            if self.snake:
                for i in [i for i in xrange(0, len(self.rows)) if i % 2]:
                    self.rows[i]= self.rows[i][::-1]
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
        print 'Stitching mosaic...'
        # Now that the canvas has been created, we can paste the
        # individual tiles on top of it. Coordinates increase from
        # (0,0) in the top left corner.
        n_row = 0  # index of row
        for row in self.rows:
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
                    try:
                        mosaic.paste(Image.open(fp.encode('cp1252')), (x, y))
                    except:
                        print ('Encountered unreadable tiles but'
                               ' could not fix them. Try installing'
                               ' ImageMagick and re-running this'
                               ' script.')
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
        # Write job parameters to file. This could be embedded in
        # the metadata.
        params = [
            self.filename,
            '-' * len(self.filename),
            'Dimensions: {}x{}'.format(self.num_cols, self.num_rows),
            'Offset within row: {}x{}'.format(self.x_offset_within_row,
                                              self.y_offset_within_row),
            'Offset between rows: {}x{}'.format(self.x_offset_between_rows,
                                                self.y_offset_between_rows),
            'Magnification: {}'.format(self.mag),
            'Snake: {}'.format(self.snake),
            '',
            #('Mosaic created using'
            #  ' MinSci Toolkit {}').format(minsci.__version__),
            #''
        ]
        fp = os.path.join(self.path, os.pardir, self.filename + '.txt')
        with open(fp, 'wb') as f:
            f.write('\n'.join(params))
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
        try:
            mosaic.create_mosaic(path)
        except NameError:
            mosaic = Mosaic(path, jpeg, skipped).create_mosaic()




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
