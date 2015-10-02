import csv
import ctypes
import decimal
import glob
import os
import re
import shutil
import sys
from textwrap import fill

import pyglet
from PIL import Image




class Selector(object):


    def __init__(self, ext='.tif'):

        decimal.getcontext().prec = 6

        root = Tkinter.Tk()
        root.withdraw()
        title = ("Please select the directory containing your tiles:")
        initial = os.path.expanduser('~')
        self.path = tkFileDialog.askdirectory(parent=root, title=title,
                                              initialdir=initial)

        tiles = [fp for fp in glob.glob(os.path.join(path, '*' + ext))]
        self.source = os.path.join(self.path, tiles)

        # Get window dimensions. These will be used to set the size of
        # the pyglet window later.
        self.window_width = ctypes.windll.user32.GetSystemMetrics(0) - 200
        self.window_height = ctypes.windll.user32.GetSystemMetrics(1) - 200




    def get_job_settings(self):
        """ Get job settings

        Attempts to read job settings from text file. Failing that,
        it will prompt the user to input that information manually.
        """
        settings = glob.glob(os.path.join(self.source, '*.apf'))
        if len(settings) == 1:
            print ('Reading job settings'
                   ' from {}...').format(os.path.basename(settings[0]))
            params = {}
            with open(settings[0], 'rb') as f:
                for line in f:
                    if line.startswith('Microscope Magnification'):
                        params['mag'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Pt:'):
                        vals = line.split(':', 1)[1].strip().split(' ')
                        if len(params) == 1:
                            key = 'ul'
                            params['z'] = vals[2]
                            params['mystery_int'] = vals[6]
                        elif len(params) == 4:
                            key = 'lr'
                        else:
                            break
                        params[key] = '{}x{}'.format(vals[0], vals[1])

        else:
            params = {}
            params['ul'] = self.prompt('Upper left:',
                                       '-?\d+(\.\d+)?x-?\d+(\.\d+)?',
                                       False)
            params['lr'] = self.prompt('Lower right:',
                                       '-?\d+(\.\d+)?x-?\d+(\.\d+)?',
                                       False)
            params['z'] = self.prompt('z:', '\d+', False)
            params['mag'] = self.prompt('Magnification:', '\d+', False)
            params['mystery_int'] = 6357060

        ul = [decimal.Decimal(x) for x in params['ul'].split('x')]
        lr = [decimal.Decimal(x) for x in params['lr'].split('x')]
        z = decimal.Decimal(params['z'])
        mag = params['mag']
        mystery_int = params['mystery_int']
        return (ul, lr, z, mag, mystery_int)






    def select(self, ul, lr, z, mag, mystery_int):
        """ Allow user to select tiles to keep """
        self.ul = ul
        self.lr = lr

        # Lighten the input tiles a bit to make the hover effect more clear
        for fp in glob.iglob(os.path.join(self.source, '*.tif')):
            im = Image.open(fp)
            mask = Image.new('L', im.size, 'white')
            im = Image.blend(im, mask, 0.75)
            im.mode = 'L'
            # Calculate coordinates
            coords = os.path.splitext(fp)[0].split('@')[1].rstrip(']')
            coords = ['{0:0>4}'.format(c) for c in coords.split(' ')[::-1]]
            coords = 'x'.join(coords)
            im.save(os.path.join(self.working, 'tile_' + coords + '.jpg'),
                    'JPEG')

        # Get sizes for the tiles and the complete moasic
        row = glob.glob(os.path.join(self.working, '*_0000x*.jpg'))
        col = glob.glob(os.path.join(self.working, '*x0000.jpg'))
        for fp in row:
            im = Image.open(fp)
            w, h = im.size
            break
        self.tiles_per_row = len(row)
        self.tiles_per_col = len(col)

        self.coordinate_width = (abs(self.ul[0] - self.lr[0]) /
                                 (self.tiles_per_row - 1))
        self.coordinate_height = (abs(self.ul[1] - self.lr[1])  /
                                  (self.tiles_per_col - 1))

        row_w = w * len(row)
        row_h = h * len(col)

        scalar_w = float(self.window_width) / row_w
        scalar_h = float(self.window_height) / row_h

        resized_w = int(w * scalar_w)
        resized_h = int(h * scalar_h)

        for fp in glob.iglob(os.path.join(self.working, '*.jpg')):
            im = Image.open(fp)
            im = im.resize((resized_w, resized_h))
            im.save(fp)

        # Open pyglet window to allow users to select tiles
        window = pyglet.window.Window(self.window_width, self.window_height)
        cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
        window.set_mouse_cursor(cursor)

        batch = pyglet.graphics.Batch()
        sprites = []
        tiles = []
        for fp in glob.iglob(os.path.join(self.working, '*.jpg')):
            im = pyglet.image.load(fp)
            y, x = os.path.splitext(fp)[0].rsplit('_', 1)[1].split('x')
            x = int(x) * im.width
            y = window.height - (int(y) + 1) * im.height
            sprites.append(pyglet.sprite.Sprite(im, x=x, y=y, batch=batch))
            tiles.append(fp)


        @window.event
        def on_draw():
            window.clear()
            batch.draw()


        @window.event
        def on_mouse_press(x, y, button, modifiers):
            for sprite in sprites:
                sprite_x, sprite_y = sprite.position
                if (sprite_x < x < sprite_x + sprite.width
                    and sprite_y < y < sprite_y + sprite.height):
                    if sprite.visible:
                        sprite.visible = False
                    else:
                        sprite.visible = True
                    break
            window.clear()
            batch.draw()


        @window.event
        def on_close():
            i = 0
            keep = []
            skip = []
            for sprite in sprites:
                if sprite.visible:
                    keep.append(tiles[i])
                else:
                    skip.append(tiles[i])
                i += 1
            # Write everything in keep to points.apf
            fp = os.path.join(self.path, 'points.apf')
            print 'Writing {}...'.format(fp)
            header = [
                    'TYPE:POINT',
                    'NUM_SEQUENCES: 1',
                    'Label:Point',
                    'Movement Mode:Points',
                    'Auto Focus:OFF',
                    'Beam Off:OFF',
                    'Microscope Magnification:' + mag + '\r\n'
                    ]
            points = []
            grid = []
            for tile in keep:
                center = self.get_center(tile)
                points.append([
                    'Pt:',
                    '{:.4f}'.format(center[0]),
                    '{:.4f}'.format(center[1]),
                    '{:.4f}'.format(z),
                    '0.0000',
                    '0.0000',
                    '0.0000',
                    mag
                    ])
            with open(fp, 'wb') as f:
                f.write('\r\n'.join(header))
                for point in points:
                    f.write(' '.join(point) + '\r\n')
                f.write('Sequence Done\r\n')
            # Copy skipped tiles for later
            for tile in skip:
                coords = os.path.splitext(tile)[0].split('_')[1].split('x')
                coords = [c.lstrip('0') if bool(c.lstrip('0'))
                          else '0' for c in coords]
                fn = '*@' + coords[1] + ' ' + coords[0] + '].tif'
                src = glob.glob(os.path.join(self.source, fn))[0]
                dst = os.path.join(self.placeholders, os.path.basename(src))
                shutil.copy2(src, dst)
            window.close()
            pyglet.app.exit()
            shutil.rmtree(self.working)
            raw_input('Done! Press any key to exit.')




        @window.event
        def on_mouse_motion(x, y, dx, dy):
            for sprite in sprites:
                sprite_x, sprite_y = sprite.position
                if (sprite_x < x < sprite_x + sprite.width
                    and sprite_y < y < sprite_y + sprite.height):
                    sprite.color = (255, 0, 0)
                else:
                    sprite.color = (255, 255, 255)

        pyglet.app.run()




    def get_center(self, tile):
        """Calculates center points for each title based on ul and lr"""
        y, x = [int(x) for x
                in os.path.splitext(tile)[0].split('_')[1].split('x')]
        center = (self.ul[0] + self.coordinate_width * x,
                  self.ul[1] - self.coordinate_height * y)
        return center




    def get_settings(self):
        """Get stored user settings

        This function sets the following parameters:
            path        location of the source files
            flag
            keep_rows
            raise_error
            add_label
        """
        # Read settings from settings.txt
        global_settings = {}
        fp = os.path.join('config', 'settings.txt')
        with open(fp, 'rb') as f:
            rows = csv.reader(f, delimiter=',', quotechar='"')
            for row in rows:
                row = [s.strip() for s in row]
                if not ''.join(row).startswith('#'):
                    key, val = row[0].split('=')
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
            open(fp, 'rb')
        except:
            # No user settings file. Create one from global file after
            # settings have been vetted.
            pass
        else:
            with open(fp, 'rb') as f:
                rows = csv.reader(f, delimiter=',', quotechar='"')
                for row in rows:
                    row = [s.strip() for s in row]
                    if not ''.join(row).startswith('#'):
                        key, val = row[0].split('=')
                        global_settings[key] = val

        # Assign settings to local variables
        yn = {'yes' : True, 'no' : False}
        params = {}
        try:
            self.path = global_settings['PATH_ROOT']
            self.flag = global_settings['OUTPUT'].lower()
            self.keep_rows = yn[global_settings['KEEP_ROWS'].lower()]
            self.raise_error = yn[global_settings['RAISE_ERROR'].lower()]
            self.add_label = yn[global_settings['ADD_LABEL'].lower()]
        except:
            raw_input('Fatal error: Settings file malformatted')
            raise
        else:
            # Test non-key variables
            if not os.path.isdir(self.path) \
               or not self.flag in ['quiet', 'debug all']:
                raw_input('Fatal error: Settings file malformatted')
                sys.exit()

        # Copy global settings file to user directory if does not exist
        try:
            open(fp, 'rb')
        except:
            src = os.path.join('config', 'settings.txt')
            dst = fp
            shutil.copy2(src, dst)




    def prompt(self, prompt, validator, confirm=True,
               helptext='No help text provided', errortext='Invalid response!'):
        """Prompts user and validates response based on validator

        Keyword arguments:
        Validator can be a str, list, or dict
        """
        # Prepare string
        prompt = '{} '.format(prompt.rstrip())
        try:
            prompt = unicode(prompt)
        except:
            pass
        # Prepare validator
        if isinstance(validator, (str, unicode)):
            validator = re.compile(validator, re.U)
        elif isinstance(validator, dict):
            prompt = '{}({}) '.format(prompt, '/'.join(validator.keys()))
        elif isinstance(validator, list):
            options = ['{}. {}'.format(x + 1, validator[x])
                       for x in xrange(0, len(validator))]
        else:
            raw_input(fill('Error in minsci.helpers.prompt: '
                           'Validator must be dict, list, or str.'))
            raise
        # Validate response
        loop = True
        while loop:
            # Print options
            if isinstance(validator, list):
                print '{}\n{}'.format('\n'.join(options), self.boundary)
            # Prompt for value
            a = raw_input(prompt).decode(sys.stdin.encoding)
            if a.lower() == 'q':
                print 'User exited prompt'
                sys.exit()
            elif a.lower() == '?':
                print '-' * 60 + '\n' + fill(helptext) + '\n' + '-' * 60
                continue
            elif isinstance(validator, list):
                try:
                    i = int(a) - 1
                    result = validator[i]
                except:
                    pass
                else:
                    if i >= 0:
                        loop = False
            elif isinstance(validator, dict):
                try:
                    result = validator[a]
                except:
                    pass
                else:
                    loop = False
            else:
                try:
                    validator.search(a).group()
                except:
                    pass
                else:
                    result = a
                    loop = False
            # Confirm value, if required
            if confirm and not loop:
                try:
                    result = unicode(result)
                except:
                    result = str(result)
                loop = self.prompt('Is this value correct: '
                                   '"{}"?'.format(result),
                                   {'y' : False, 'n' : True},
                                   confirm=False)
            elif loop:
                print '-' * 60 + '\n' + fill(errortext) + '\n' + '-' * 60
        # Return value as unicode
        return result


# Handle user input
selector = Selector()
params = selector.get_job_settings()
selector.select(*params)
