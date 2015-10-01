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
try:
    from PIL import Image
except:
    from pillow import Image

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
        self.path = path
        self.extmap = {
            '.jpg' : 'JPEG',
            '.tif' : 'TIFF',
            '.tiff' : 'TIFF'
        }




    def stitch(self):
        """Calls parameters in proper order"""
        self.set_parameters()
        self.classify_tiles()
        self.list_tiles()




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
        self.dim = [key for key in dims
                    if dims[key] == max(dims.values())][0]
        self.w, self.h = [int(x) for x in self.dim.split('x')]
        self.name = os.path.dirname(path)
        self.tiles = self.sort_tiles([tile for tile in tiles
                                      if tile.endswith(self.ext)])
        return self




    def set_mosaic_parameters(self):
        """Prompt user for job parameters"""
        yes_no = {'y' : True, 'n' : False}
        try:
            self.num_cols
        except UnboundLocalVariable:
            self.num_cols = int(prompt('Number of columns:', '\d+'))
        self.mag = int(prompt('Magnification:', '\d+'))
        self.snake = prompt('Snake pattern?', yes_no)
        if not prompt('Are these parameters okay?', yes_no):
            self.set_parameters()
        else:
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
            if ' ' in key:
                x, y = key.split(' ')
                i = int(y) * self.num_cols + int(x)
                cols.append(y)
            else:
                i = int(key)
            temp[i] = tile
        # Bonus: Determine the number of columns if the tiles are
        # provided with SEM-style grid notation. This is kind of an
        # odd fit here, but I don't know where else to put it.
        try:
            self.num_cols = max(cols) + 1
        except UnboundLocalVariable:
            pass
        return [temp[key] for key in sorted(temp.keys())]






    def create_mosaic(self):
        self.rows = self.mandolin(tiles, self.num_cols)
        self.num_rows = len(self.rows)
        # The dimensions of the mosaic are determined by the
        # tile dimensions MINUS the offset within the row
        # PLUS the offset between rows.
        x_offset_within_row = 10
        x_offset_between_rows = 10
        mosaic_width = ((self.w - x_offset_within_row)  * self.num_cols +
                        x_offset_between_rows * self.num_rows)
        y_offset_within_row = 10
        y_offset_between_rows = 10
        mosaic_height = ((self.h + y_offset_within_row) * self.num_cols +
                         y_offset_between_rows * self.num_rows
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height))
        # Now that the canvas has been created, we can paste the
        # individual tiles on top of it. We need to pay attention
        # to the direction of the offsets.
        y = 0
        for row in self.rows:
            if self.snake and not (y + 1) % 2:
                row = row[::-1]
            x = 0
            for fp in row:
                if bool(fp):
                    mosaic.paste(Image.open(fp), (self.w * x, self.h * y))
                x += 1
            y += 1
        mosaic.save(self.name + self.ext, self.extmap[self.ext])





    def determine_offset(self, same_row=True):
        """Use pyglet to allow users to set offset between tiles"""
        # Offsets are defined as follows:
        #  Within row: y is positive if the top edge of the right
        #   tile is HIGHER than that of the left (stair step up).
        #   Because there is overlap, x is always positive.
        #  Between rows: x is positive if the left edge of the lower
        #   tile is to the RIGHT of the upper tile. Because there is
        #   overlap, y is always positive.
        y = self.num_rows / 2
        x = self.num_cols / 2
        try:
            t1 = self.rows[y][x]
            t2 = self.rows[y][x]
        except:
            pass
        if same_row:
            img = Image.new('RGB', (self.w * 2, self.h * 1.1))
            img.paste(Image.open(f1), (0, 0))
            img.paste(Image.open(f2), (self.w, 0))
        else:
            img = Image.new('RGB', (self.w * 1.1, self.h * 2))
            img.paste(Image.open(f1), (0, 0))
            img.paste(Image.open(f2), (0, self.h))
        # Resize image to fit in the current screen
        img.resize()
        pass




    def mandolin(self, lst, n):
        """Split list into groups of n members

        @param list
        @param int
        @return list
        """
        return [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]




    def patch_tiles(self, tiles):
        """Patch tileset with blanks based on mosaic.Selector"""
        pass



'''


def custom_setting(path):
    #remove any existing mpc/cache files from current directory
    int_exts = ('*.mpc', '*.cache')
    files = []
    for int_ext in int_exts:
        files.extend(glob.glob(os.path.join(path, int_ext)))
    for f in files:
        os.remove(f)
    #notify user of ability to review
    print '-' * 60 + '\n' +\
          'You will have an opportunity to review these settings before ' +\
          'image processing begins. Type ? for help.\n' +\
          '-' * 60
    #prompt user for number of columns
    loop = True
    while loop:
        num_cols = raw_input('Number of columns: ')
        if num_cols.isdigit():
            print 'The raster contains ' + num_cols + ' columns'
            loop = False
        elif num_cols == '?':
            print '-' * 60 + '\nNUMBER OF COLUMNS\n' +\
                  'Only the number of columns must be specified. ' +\
                  'The number of rows is determined automatically.\n' +\
                  '-' * 60
        else:
            print 'Invalid value!'
    #prompt user for raster info
    print '-' * 60
    loop = True
    while loop:
        mag = raw_input('Magnification (ex. 2.5): ')
        if is_number(mag):
            print 'The images were collected at ' + mag + 'x magnification'
            loop = False
        elif mag == '?':
            print '-' * 60 + '\nMAGNIFICATION\n' +\
                  'Magnification used to collected images.' +\
                  '-' * 60
        else:
            print 'Invalid value!'
    #prompt user for pattern type
    print '-' * 60
    loop = True
    while loop:
        a = raw_input('Is this a snake pattern? (y/n) ')
        if a.lower() == 'y':
            snake = 'True'
            loop = False
        elif a.lower() == 'n':
            snake = 'False'
            loop = False
        elif a == '?':
            print '-' * 60 + '\nSNAKE PATTERN\n' +\
                  'Yes if snake pattern, no if raster.\n' +\
                  '-' * 60
        else:
            print 'Invalid input!'
    #get dimensions and extension type for tiles
    if snake == 'True':
        rows = get_rows(path, int(num_cols), True)
    else:
        rows = get_rows(path, int(num_cols), False)
    img = rows[1]
    ext = rows[2]
    rows = rows[0]
    #get within-row offset using offset.py if more than one col
    if len(rows[0]) > 1:
        print '-' * 60
        offset_col = offset.get_offset(path, rows, img, '', True)
        print 'Within-row offset is ' + offset_col + ' pixels'
    else:
        offset_col = '0x0'
    #get between-row offset using offset.py if more than one row
    if len(rows) > 1:
        print '-' * 60
        offset_row = offset.get_offset(path, rows, img, '', False)
        print 'Raw between-row offset is ' + offset_row + ' pixels'
    else:
        offset_row = '0x0'
    #correct between-row offset for within-row offset
    #used only if rows stair-step down
    if int(offset_col.split('x')[1]) > 0:
        print 'Applying correction to account for within-row offset...'
        offset_y_col = offset_col.split('x')[1]
        offset_fix = offset_row.split('x')
        offset_y_row = int(offset_fix[1]) +\
                       int(offset_y_col) * (int(num_cols) - 1)
        offset_row = offset_fix[0] + 'x' + str(offset_y_row)
        print 'Corrected between-row offset is ' + offset_row + ' pixels'
    #remove temp file
    os.remove(os.path.join(path, 'temp.jpg'))
    #return values as list
    return [num_cols, img, offset_col, offset_row, snake, '', mag]


def enq(s):
    #adds quotes to string
    return '"' + s + '"'


def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


def sorter(s):
    #parses filename for string to sort on
    #add new parsings by appending the the global regex list
    global regex
    for r in regex:
        p = r[0] #regex pattern
        row_first = r[1] #true if row is first match
        try:
            c = p.search(s)
            if row_first:
                row_num = c.group(1)
                col_num = c.group(2)
            else:
                row_num = c.group(2)
                col_num = c.group(1)
        except:
            if c:
                return '{0:0>4}'.format(c.group(1))
        else:
            return '{0:0>4}'.format(row_num) +\
                   '{0:0>4}'.format(col_num)
    #return original string if no matches
    return s


def scale_parameter(param, scalar):
    return int(round(float(scalar) * param))


def get_rows(path, num_cols, snake):
    #create mode dictionary
    modes = {
        'L' : 8,
        'P' : 8,
        'RGB' : 8,
        'RGBA' : 8
        }
    #create param dict to store  image info
    p = {
        'ext' : {},
        'dim' : {},
        'bits' : {}
        }
    #notify user
    print '-' * 60 + '\nDirectory is ' + path
    d = os.path.basename(path)
    #compile regex
    re_rows = re.compile('r\d+[.]')
    #get image parameters
    for f in os.listdir(path):
        #get extension for each file
        ext = os.path.splitext(f)[1]
        try:
            p['ext'][ext] = p['ext'][ext] + 1
        except:
            p['ext'][ext] = 1
        #get dimensions and pixel depth
        try:
            im = Image.open(os.path.join(path, f))
        except:
            #image could not be opened
            pass
        else:
            #get dimensions
            dim = im.size
            dim = str(dim[0]) + 'x' + str(dim[1])
            try:
                p['dim'][dim] = p['dim'][dim] + 1
            except:
                p['dim'][dim] = 1
            #get pixel depth (should be 8 or 16)
            mode = im.mode
            try:
                bits = modes[mode]
            except:
                pass
                print 'Mode ' + mode + ' does not exist'
            else:
                try:
                    p['bits'][bits] += 1
                except:
                    p['bits'][bits] = 1
    #find most common extension
    extensions = p['ext'].items()
    extensions.sort(key = lambda extension: extension[1])
    in_ext = extensions.pop()[0]
    #find most common dimensions
    dimensions = p['dim'].items()
    dimensions.sort(key = lambda dimension: dimension[1])
    in_dim = dimensions.pop()[0]
    #find most common depth
    bits = p['bits'].items()
    bits.sort(key = lambda bit: bit[1])
    in_bits = bits.pop()[0]
    #get coordinates from files matching extension
    coordinates = []
    for f in patch_mosaic(path, ext, num_cols):
        #split off name
        base = os.path.splitext(os.path.basename(f))[0]
        #exclude row files and directory
        if re_rows.match(base) or base == d:
            os.remove(f)
        elif f != path:
            try:
                #append filename and sort value
                c = re_sort.search(base).group()
                coordinates.append([f, c])
            except:
                #assume alphabetical sort
                coordinates.append([f, 0])
    #coordinates.sort(key=len)
    #coordinates.sort(key=lambda c: sorter(c[1]))
    rows = [[]]
    i = 1
    for coordinate in coordinates:
        f = coordinate[0]
        if i % (num_cols + 1) == 0:
            #print 'New row appended at ' + coordinate[4]
            rows.append([]) #add new row as empty list
            i = 1 #reset counter
        row = len(rows) - 1
        rows[row].append(f)
        i += 1
    #reverse alternate rows if snake
    temp = []
    i = 0
    for row in rows:
        if snake and not i % 2 == 0 and i > 0:
            row.reverse()
        temp.append(row)
        i += 1
    rows = temp
    #return rows and ext as list
    return [rows, in_dim, in_ext, in_bits]


def scale_from_mag(mag, kind):
    #scalebar properties
    px_per_inch = None
    if 'petrographic' in kind:
        if mag == 2.5:
            px_per_inch = 8500
    #return pixels per mm, if known
    if px_per_inch:
        px_per_mm = px_per_inch / 25.4
        return px_per_mm
    else:
        return None


def draw_label(this_file, w_full, h_full):
    global scope_type
    #scale label based on area of mosaic
    # Fonts: 10000x10000 = 144pt
    # Scale: 10000x10000 = 80
    area = w_full * h_full
    ref_dim = 10000
    ref_area = ref_dim**2
    #font_size = long(area * 144 / ref_area)
    h_rec = long(area * 100 / ref_area)
    font_size = long(h_full * 144 / ref_dim)
    #h_rec = long(h_full * 80 / ref_dim)
    #return list with label and scalebar dimensions
    p = os.path.basename(this_file).split('_')
    number = p[0]
    mag = float(p[2].replace('mag','').replace('x',''))
    #process suffix for type
    re_num = re.compile('[0-9]+')
    re_let = re.compile('[A-z]+')
    #get suffix
    if re_num.match(p[1]):
        suffix = '-' + re_num.match(p[1]).group(0)
    else:
        suffix = ''
    #get image type
    try:
        key = re_let.search(p[1]).group(0)
        kind = scope_type[key] + ', '
    except:
        kind = 'unspecified image type'
    #scalebar properties
    px_per_mm = scale_from_mag(mag, kind)
    x1 = w_full - 600 #starting x-coordinate of scalebar
    y1 = h_full - 160 #starting y-coordinate of scalebar
    #create a scalebar if possible
    if px_per_mm:
        w_rec = px_per_mm
        x2 = x1 + w_rec
        y2 = y1 + h_rec
        cmd = ['-pointsize',
               str(font_size),
               '-annotate',
               '+' + str(120) + '+' + str(y1 + 80),
               #enq('NMNH ' + number + suffix +\
               enq(number + suffix +\
                   ' (' + kind.strip(' ').strip(',') + ')'),
               #enq('NMNH ' + number + suffix + ' (' + kind +\
               #    str(int(w_full / px_per_mm)) + ' mm wide)'),
               '-annotate',
               '+' + str(x1 + 100) + '+' + str(y1 + 80),
               enq('5 mm'),
               '-fill',
               'black',
               '-draw',
               enq('rectangle ' +\
                   str(long(x1 - w_rec * 5)) + ',' + str(y1 - 1) + ' ' +\
                   str(x1) + ',' + str(y1 - 5)),
               '-draw',
               enq('rectangle ' +\
                   str(long(x1 - w_rec * 5)) + ',' + str(y2 + 1) + ' ' +\
                   str(x1) + ',' + str(y2 + 5))]
        i = 1
        while i <= 5:
            if i % 2 == 0:
                bg_color = 'white'
            else:
                bg_color = 'black'
            x1 -= px_per_mm
            x2 -= px_per_mm
            cmd.append('-fill')
            cmd.append(bg_color)
            cmd.append('-draw')
            cmd.append(enq('rectangle ' +\
                           str(x1) + ',' + str(y1) + ' ' +\
                           str(x2) + ',' + str(y2)))
            i += 1
    #label, no scalebar
    else:
         cmd = ['-pointsize',
               str(font_size),
               '-annotate',
               '+' + str(120) + '+' + str(y1 + 80),
               enq('' + number + suffix +\
                   ' (' + kind.strip(' ').strip(',') + ')')]
    #return command
    return cmd


def clean_fn(fn):
    #remove underscorces
    if fn.count('_') > 1:
        arr = fn.split('_')
        fn = ''.join(arr[:-1]) + '_' + arr[-1]
    #add underscore if missing
    if not '_' in fn:
        fn += '_unspecified'
    #remove invalid or unwanted characters
    fn = fn.replace(' ', '')
    #return clean filename
    return fn


def patch_mosaic(path, ext, num_cols):
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


#-------------------------------------------------------------------------------
#-PROGRAM LOGIC-----------------------------------------------------------------
#-------------------------------------------------------------------------------
raise_error = False
try:
    # Convert yes/no from configuration files to boolean
    yes_no_to_bool = {
        'yes' : True,
        'no' : False
        }

    # Read settings from settings.txt
    global_settings = {}
    fp = os.path.join('config', 'settings.txt')
    with open(fp, 'rb') as f:
        rows = csv.reader(f, delimiter=',', quotechar='"')
        for row in rows:
            row = [s.strip() for s in row]
            if not ''.join(row).startswith('#'):
                arr = row[0].split('=')
                key = arr[0]
                val = arr[1]
                global_settings[key] = val

    # Try to read settings from user's custom file
    user = os.environ.get("USERNAME")
    try:
        os.makedirs(os.path.join('config', 'users', user))
    except:
        # User directory already exists
        pass
    fp = os.path.join('config', 'users', user, 'settings.txt')
    try:
        with open(fp, 'rb'): pass
    except:
        # No user settings file. Create one from global file after
        # settings vetted.
        pass
    else:
        with open(fp, 'rb') as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            first = True
            for row in rows:
                row = [s.strip() for s in row]
                if not ''.join(row).startswith('#'):
                    arr = row[0].split('=')
                    key = arr[0]
                    val = arr[1]
                    global_settings[key] = val

    # Assign settings to local variables
    try:
        base_dir = global_settings['PATH_ROOT']
        flag = global_settings['OUTPUT'].lower()
        keep_rows = yes_no_to_bool[global_settings['KEEP_ROWS'].lower()]
        raise_error = yes_no_to_bool[global_settings['RAISE_ERROR'].lower()]
        add_label = yes_no_to_bool[global_settings['ADD_LABEL'].lower()]
    except:
        raw_input('Fatal error: Settings file')
        raise
    else:
        # Test non-key variables
        if not os.path.isdir(base_dir) \
           or not flag.lower() in ['quiet', 'debug all']:
            raw_input('Fatal error: Settings file')
            sys.exit()

    # Copy global settings file to user directory if does not exist
    try:
        with open(fp, 'rb'): pass
    except:
        src = os.path.join('config', 'settings.txt')
        dst = fp
        shutil.copy2(src, dst)


    # Read regex for use in sorter function from file_sort_regex.txt
    regex = []
    fp = os.path.join('config', 'file_sort_regex.txt')
    try:
        with open(fp, 'rb'): pass
    except:
        raw_input('Fatal error: Regex file')
        raise
    else:
        with open(fp, 'rb') as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            i = 0
            for row in rows:
                row = [s.strip() for s in row]
                if not ''.join(row).startswith('#'):
                    regex.append([row[0], yes_no_to_bool[row[1].lower()]])

    # Create sort matching function and compile strings
    re_sort = ''
    temp = []
    for r in regex:
        re_sort += '|' + r[0]
        temp.append([re.compile(r[0]), r[1]])
    re_sort = re.compile(re_sort.strip('|'))
    regex = temp

    #scope_type for use in draw_label function
    scope_type = {}
    fp = os.path.join('config', 'image_types.txt')
    try:
        with open(fp, 'rb'): pass
    except:
        raw_input('Fatal error: Scope type file')
        raise
    else:
        with open(fp, 'rb') as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            i = 0
            for row in rows:
                row = [s.strip() for s in row]
                if not ''.join(row).startswith('#'):
                    key = row[0]
                    val = row[1]
                    scope_type[key] = val

    #set working variables
    #working_dir = os.path.join(base_dir, 'Workflows', 'Mosaics')
    working_dir = base_dir
    print 'Working directory set to ' + working_dir

    #get saved settings from config/settings
    settings = []
    for line in open(os.path.join('config', 'processes.txt'), 'rb'):
        settings.append(line.strip().split(';'))
    len_global = len(settings)

    #get saved settings from user's custom file
    try:
        f = open(os.path.join('config', 'users', user, 'processes.txt'), 'rb')
    except:
        try:
            os.makedirs(os.path.join('config', 'users', user))
        except:
            # Directory already exists
            pass
        #create custom settings file if does not exist
        open(os.path.join('config', 'users', user, 'processes.txt'), 'w')
        f = []
        print '-' * 60 + '\nCreated custom settings file for ' + user +\
              '\n' + '-' * 60
    for line in f:
        settings.append(line.strip().split(';'))

    #add settings to user settings
    settings.append(['Remove user-defined settings'])

    #notify user about settings
    if flag:
        print '>> ' + flag.capitalize() + ' mode enabled'
    if keep_rows:
        print '>> Intermediate files will be saved'
    if raise_error:
        print '>> Show traceback on error (use only with IDLE)'
    if add_label:
        print '>> Will try to add label and scalebar'

    #select setting to use
    loop = True
    while loop:
        #print list of saved settings
        print '-' * 60 + '\nOPTIONS:'
        i = 1
        for setting in settings:
            print '(' + str(i) + ') ' + setting[0]
            i += 1
        print '-' * 60
        c = raw_input('Select by number: ')
        if c.isdigit() and 0 < int(c) <= (len(settings) - 1):
            c = int(c)
            #pull from saved values
            if c > 1:
                val = settings[c -1]
                out_ext = '.tif' #output file extension
                row_ext = '.tif' #intermediate file extension
                calibrating = False #all rows, will not preserve intermediates
            #calibrate using custom_setting()
            elif c == 1:
                path = os.path.join(working_dir)
                #choose a directory for calibration
                dirs = os.listdir(path)
                for d in dirs:
                    if os.path.basename(d) != 'output' and +\
                    os.path.isdir(os.path.join(path, d)):
                        break
                path = os.path.join(path, d)
                val = custom_setting(path)
                val = ['Calibration'] + val
                out_ext = '.tif' #output file extension
                row_ext = '.tif' #intermediate file extension
                calibrating = True #first two rows, will preserve intermediates
            #print values for review
            print '-' * 60
            print 'You\'ve selected the following parameters:'
            print '  Name:                ' + val[0]
            print '  Numbers of columns:  ' + val[1]
            print '  Magnification:       ' + val[7]
            print '  Snake pattern:       ' + val[5]
            #print '  Normalize grays:     ' + val[6]
            print '  Image resolution:    ' + val[2]
            print '  Within-row offset:   ' + val[3]
            print '  Between-row offset:  ' + val[4]
            print '-' * 60
            #ask user to confirm values
            loop2 = True
            while loop2:
                c2 = raw_input('Are these values correct? (y/n) ')
                if c2.lower() == 'y':
                    loop = False
                    loop2 = False
                elif c2.lower() == 'n':
                    loop2 = False
                else:
                    'Invalid inpput!'
            #assign parameters to local variables
            num_cols = int(val[1]) #number of columns
            w = int(val[2].split('x')[0]) #width of tile
            h = int(val[2].split('x')[1]) #height of tile
            x_col = int(val[3].split('x')[0]) #x offset, same row
            y_col = int(val[3].split('x')[1]) #y offset, same row
            x_row = int(val[4].split('x')[0]) #x offset, same col
            y_row = int(val[4].split('x')[1]) * -1 #y offset, same col
            mag = val[7] #magnification
            #convert snake to boolean
            if (val[5] == 'True'):
                snake = True
            else:
                snake = False
            #convert gamma to boolean
            if (val[6] == 'True'):
                gamma = True
            else:
                gamma = False
        #allow user to remove one or all customized settings
        elif c.isdigit() and int(c) == len(settings):
            loop2 = True
            while loop2:
                c2 = raw_input('Enter number you\'d like to delete '+\
                               '(type A to delete all user settings) : ')
                if c2 == 'A':
                    try:
                        #remove all user values from settings list
                        settings = settings[:len_global]
                        #clear settings file
                        f = open(os.path.join('config', user), 'w')
                        f.close()
                    except:
                        #f does not exist, so no data to clear
                        pass
                    loop2 = False
                elif c2.isdigit()\
                     and len_global < int(c2) < len(settings):
                    #remove selected index from both user_settings and settings
                    settings.pop(int(c2) - 1)
                    #create slice corresponding to user settings
                    n = len_global
                    m = len(settings) - 1
                    user_settings = settings[n:m]
                    #write user settings back to file
                    f = open(os.path.join('config', user), 'w')
                    f.write('\n'.join([';'.join(arr) for arr in user_settings]))
                    f.close()
                    loop2 = False
                else:
                    print 'Invalid input!'
        else:
            print 'Invalid input! Response must be a number between 1 and ' +\
                  str(len(settings))

    #check if manual
    if c == 1:
        print 'You will have an opporunity to save your settings once the ' +\
              'script has finished running.'
        manual = val
    else:
        manual = []

    #calculate dimensions of row file
    w_row = w + (w + x_col) * (num_cols - 1)
    h_row = h + abs(y_col) * (num_cols - 1)
    print 'Each row will be ' +\
          str(w_row) + ' pixels wide and ' +\
          str(h_row) + ' pixels high'

    #hide command window
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    #get imagemagick convert command
    convert = 'imconvert'

    #create output directory if it does not exist, clear it if it does
    try:
        os.makedirs(os.path.join(working_dir, 'output'))
    except:
        for root, dirs, files in os.walk(os.path.join(working_dir, 'output')):
            for f in files:
                if not 'placeholders' in root:
                    os.unlink(os.path.join(root, f))
            for d in dirs:
                if not d == 'placeholders':
                    shutil.rmtree(os.path.join(root, d))
    os.makedirs(os.path.join(working_dir, 'output', 'working'))

    #intermediate filetypes
    int_exts = ('*.mpc', '*.cache')

    #walk through mosaic directories
    for root, dirs, files in os.walk(working_dir):
        #empty output directory
        if 'output' in dirs:
            dirs.remove('output')
        for d in [d for d in dirs]:
            #set start time
            start_time = datetime.now()
            #get current directory
            current_dir = os.path.join(working_dir, d)
            #remove any existing mpc/cache files from current directory
            files = []
            for int_ext in int_exts:
                files.extend(glob.glob(os.path.join(current_dir, int_ext)))
            for f in files:
                #print f
                os.remove(f)
            #create row list
            temp = get_rows(current_dir, int(num_cols), snake)
            rows = temp[0]
            in_ext = temp[2]
            depth = temp[3]
            #set coordinate baselines to account for negative offsets
            if x_row >= 0:
                x_base = 0
            else:
                x_base = abs(x_row) * len(rows)
            if y_col >= 0:
                y_base = 0
            else:
                y_base = abs(y_col) * num_cols
            #notify user about mosaic dimensions
            w_full = w_row + abs(x_row) * (len(rows) - 1)
            h_full = h_row + (h_row + y_row - y_base) * (len(rows) - 1) + 250
            print 'Final mosaic will be ' +\
                  str(w_full) + ' px wide and ' +\
                  str(h_full) + ' px high'
            ####################################################################
            # BEGIN MEMORY ADJUSTMENT
            ####################################################################
            #check mosaic size against system memory
            #The working copy of an image used by ImageMagick must be
            #written to RAM or it will take forever to finish. You can
            #estimate the memory used by multiplying AREA IN PIXELS by
            #BYTES PER PIXEL, then converting to GB (1024**3 bytes?).
            print '-' * 60 + '\nChecking if mosaic will fit in memory...'
            mem_available = 0.8 * psutil.virtual_memory().available/\
                            1024**3 #available RAM in GB
            mem_available = 100
            #Cannot get the coefficient to stabilize. Max filesize is 2GB.
            mem_image = float(1) * w_full * h_full * depth / 1024**3
            if mem_image > 4 or mem_image >= mem_available:
                print 'The mosaic is too big!'
                if mem_image > 4 and mem_available >= 4:
                    print 'The full image would be ' + str(round(mem_image, 1)) +\
                          ' GB but TIFFs must be 4 GB or less'
                    scalar = sqrt(3.9 / mem_image)
                else:
                    print 'The full image requires ' + str(round(mem_image, 1)) +\
                          ' GB for processing but only ' +\
                          str(round(mem_available, 1)) + ' GB is available'
                    scalar = sqrt(mem_available / mem_image)
                #resize to viable dimensions using scalar
                print 'Resizing mosaic to fit...'
                w = scale_parameter(w, scalar)
                h = scale_parameter(h, scalar)
                w_row = scale_parameter(w_row, scalar)
                h_row = scale_parameter(h_row, scalar)
                w_full = scale_parameter(w_full, scalar)
                h_full = scale_parameter(h_full, scalar)
                x_col = scale_parameter(x_col, scalar)
                y_col = scale_parameter(y_col, scalar)
                x_row = scale_parameter(x_row, scalar)
                y_row = scale_parameter(y_row, scalar)
                #reset coordinate baselines
                if x_row >= 0:
                    x_base = 0
                else:
                    x_base = abs(x_row) * len(rows)
                if y_col >= 0:
                    y_base = 0
                else:
                    y_base = abs(y_col) * num_cols
                print 'Resized mosaic will be ' +\
                      str(w_full) + ' px wide and ' +\
                      str(h_full) + ' px high\n' +\
                      '-' * 60
            ####################################################################
            # END MEMORY ADJUSTMENT
            ####################################################################
            #create rows
            print '-' * 60 + '\nWorking...\n  Creating rows...'
            n = 0
            for row in rows:
                num = '0' * (4 - len(str(n))) + str(n)
                this_row = os.path.join(working_dir, 'output', 'working',
                                        'r' + '0' * (4 - len(str(n))) +\
                                        str(n) + row_ext)
                #create row command
                cmd = [convert,
                       '-size',
                       str(w_row) + 'x' + str(h_row),
                       'xc:white']
                #set flag
                if flag:
                    cmd.append('-' + flag)
                #add tiles to command
                i = 0
                while i < len(row):
                    #set filename with path
                    this_file = os.path.join(current_dir, row[i])
                    #get offset in x
                    x_offset = (w + x_col) * i
                    if x_offset >= 0:
                        this_offset = '+' + str(x_offset)
                    else:
                        this_offset = str(x_offset)
                    #get offset in y
                    y_offset = y_base + y_col * (num_cols - i)
                    if y_offset >= 0:
                        this_offset += '+' + str(y_offset)
                    else:
                        this_offset += str(y_offset)
                    #set this_geometry
                    this_geometry = str(w) + 'x' + str(h) + this_offset
                    #append to command
                    this_cmd = [enq(this_file),
                                '-geometry',
                                this_geometry,
                                '-composite']
                    cmd += this_cmd
                    i += 1
                #add row file to command
                cmd += ['-transparent',
                        'white',
                        enq(this_row)]
                #execute row command
                #print ' '.join(cmd)
                cmd = shlex.split(' '.join(cmd))
                subprocess.call(cmd, startupinfo=startupinfo)
                #iterate row number
                n += 1
            #cmd = ['mogrify',
            #       '-depth',
            #       '8',
            #       enq(os.path.join(working_dir, 'output', 'working', '*.tif'))]
            #print ' '.join(cmd)
            #subprocess.call(cmd, startupinfo=startupinfo)
            #raw_input()
            #create mosaic by placing row files on a blank image
            #raw_input()
            print '  Creating mosaic...'
            this_full = clean_fn(d) + \
                        '_mag' + str(mag) + 'x' +\
                        '_dim' + str(num_cols) + 'x' + str(len(rows)) +\
                        '_col' + str(x_col) + 'x' + str(y_col) + \
                        '_row' + str(x_row) + 'x' + str(y_row)
            if snake:
                this_full += '_snake'
            else:
                this_full += '_raster'
            this_full = os.path.join(working_dir, 'output', this_full + out_ext)
            cmd = [convert,
                   '-size',
                   str(w_full) + 'x' + str(h_full),
                   'xc:white']
            #set flag
            if flag:
                cmd.append('-' + flag.lower())
            #iterate through row files
            i = 0
            for f in glob.iglob(os.path.join(working_dir, 'output',
                                             'working','*' + row_ext)):
                #get offset in x
                x_offset = x_base + x_row * (i + 1)
                if x_offset >= 0:
                    this_offset = '+' + str(x_offset)
                else:
                    this_offset = str(x_offset)
                #get offset in y
                y_offset = (h_row + y_row - y_base) * i
                if y_offset >= 0:
                    this_offset += '+' + str(y_offset)
                else:
                    this_offset += str(y_offset)
                #append to command
                this_cmd = [enq(f),
                            '-geometry',
                            this_offset,
                            '-composite']
                cmd += this_cmd
                #iterate row counter
                i += 1
            if add_label:
                cmd += draw_label(this_full, w_full, h_full)
            cmd += [enq(this_full)]
            #print ' '.join(cmd)
            cmd = shlex.split(' '.join(cmd))
            subprocess.call(cmd, startupinfo=startupinfo)
            #check that mosaic exists
            try:
                open(this_full, 'rb')
            except:
                print '  Error: Creation of mosaic failed'
            #finish up if creation of mosaic is successful
            else:
                print '  Creating jpeg...'
                cmd = [convert,
                       enq(this_full),
                       '-sample',
                       '50x50%',
                       '-format',
                       'jpg',
                       enq(os.path.splitext(this_full)[0] + '.jpg')]
                #print ' '.join(cmd)
                cmd = shlex.split(' '.join(cmd))
                subprocess.call(cmd, startupinfo=startupinfo)
                #check that mosaic jpeg exists
                try:
                    open(os.path.splitext(this_full)[0] + '.jpg', 'rb')
                except:
                    print '  Error: Creation of jpeg failed'
                #deleting row files
                if keep_rows:
                    print '  Moving intermediate files...'
                    #create row directory if does not exist
                    try:
                        os.mkdir(os.path.join(working_dir, 'output', 'rows'))
                    except:
                        pass
                    #create specfic directory for current mosaic
                    os.mkdir(os.path.join(working_dir, 'output', 'rows', d))
                    #move row files to mosaic directory
                    for f in glob.iglob(os.path.join(working_dir, 'output',
                                                     'working', '*' + row_ext)):
                        src = os.path.join(working_dir, 'output','working', f)
                        dst = os.path.join(working_dir, 'output', 'rows', d)
                        shutil.move(src, dst)
                else:
                    print '  Deleting intermediate files...'
                    for f in glob.iglob(os.path.join(working_dir, 'output',
                                                     'working', '*' + row_ext)):
                        os.remove(f)
                #notify user mosaic is complete
                t = datetime.now() - start_time
                print 'Mosaic complete!\n' +\
                      this_full + '\n' +\
                      'Stitching time was ' + str(t)
    #delete working directory
    try:
        os.rmdir(os.path.join(working_dir, 'output', 'working'))
    except:
        print 'Could not delete working directory'
    #offer user the chance to save manual settings
    if len(manual) > 0:
        print '-' * 60
        loop = True
        while loop:
            c = raw_input('Would you like to save your settings? (y/n) ')
            if c.lower() == 'y':
                name = raw_input('Provide a name for these settings: ')
                val[0] = name
                o = open(os.path.join('config', 'users', user, 'processes.txt'),
                         'a')
                o.write('\n' + ';'.join(val))
                o.close()
                loop = False
            elif c.lower() == 'n':
                loop = False
            else:
                'Invalid input!'
except KeyError:
    raise
except:
    if raise_error:
        raise
    else:
        raw_input('-' * 60 + '\n' +\
                  'The script stopped unexpectedly. ' +\
                  'Press any key to close this window.')
else:
    #notfy user that script completed successfully
    print '-' * 60
    raw_input('Done! Press any key to exit.')
'''
