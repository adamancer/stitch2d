import decimal
import glob
import os
import shutil
import sys
import time
import Tkinter
import tkFileDialog
from copy import copy
from textwrap import fill

import pyglet
import pyglet.window
from PIL import Image

from .mosaic import mandolin
from ..helpers import prompt




class Selector(object):


    def __init__(self, path=None, ext='.tif'):

        self.ext = ext

        decimal.getcontext().prec = 6

        if not path:
            root = Tkinter.Tk()
            root.withdraw()
            title = ("Please select the directory containing your tiles:")
            initial = os.path.expanduser('~')
            self.source = tkFileDialog.askdirectory(parent=root, title=title,
                                                    initialdir=initial)
        else:
            self.source = path
        print 'Source is {}'.format(self.source)

        # Reintegrate previously skipped files if they exist
        try:
            os.remove(os.path.join(self.source, 'skipped.txt'))
        except IOError:
            pass
        try:
            os.remove(os.path.join(self.source, 'screenshot.jpg'))
        except IOError:
            pass
        try:
            path = os.path.join(self.source, 'skipped', '*' + self.ext))
            skipped = [fp for fp in glob.glob(path)]
        except IOError:
            pass
        else:
            for src in skipped:
                dst = os.path.join(self.source)
                shutil.move(src, dst)

        tiles = [fp for fp in glob.glob(os.path.join(self.source,
                                                     '*' + self.ext))]

        # Get window dimensions. These will be used to set the size of
        # the pyglet window later.
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        self.window_width = screen.width - 200
        self.window_height = screen.height - 200




    def get_job_settings(self):
        """ Get job settings

        Attempts to read job settings from text file. Failing that,
        it will prompt the user to input that information manually.
        """
        settings = glob.glob(os.path.join(self.source, '*.apf'))
        if len(settings):
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
            params['ul'] = prompt('Upper left:',
                                  '-?\d+(\.\d+)?x-?\d+(\.\d+)?',
                                  False)
            params['lr'] = prompt('Lower right:',
                                  '-?\d+(\.\d+)?x-?\d+(\.\d+)?',
                                  False)
            params['z'] = prompt('z:', '\d+', False)
            params['mag'] = prompt('Magnification:', '\d+', False)
            params['mystery_int'] = 6357060

        ul = [decimal.Decimal(x) for x in params['ul'].split('x')]
        lr = [decimal.Decimal(x) for x in params['lr'].split('x')]
        z = decimal.Decimal(params['z'])
        mag = params['mag']
        mystery_int = params['mystery_int']
        self.ul = ul
        self.lr = lr
        return (ul, lr, z, mag, mystery_int)




    def select(self, ul, lr, z, mag, mystery_int):
        """ Allow user to select tiles to keep """
        print 'Preparing selection grid...'

        # Lighten the input tiles a bit to make the hover effect more clear
        cols = []
        grid = {}
        for fp in glob.iglob(os.path.join(self.source, '*' + self.ext)):
            img = Image.open(fp)
            mask = Image.new('L', img.size, 'white')
            img = Image.blend(img, mask, 0.75)
            img.mode = 'L'
            # Calculate coordinates
            try:
                key = os.path.splitext(fp)[0].split('@')[1].rstrip(']')
            except IndexError:
                raw_input("Malformatted filename: Check folder for"
                          " images that aren't part of the grid")
                raise
            x, y = key.split(' ')
            cols.append(int(x))
            w, h = img.size
            grid[key] = (copy(img), fp)

        self.num_cols = max(cols) + 1
        self.num_rows = len(grid) / self.num_cols

        filenames = {}
        for key in grid.keys():
            x, y = key.split(' ')
            i = int(x) + self.num_cols * int(y)
            grid[i] = grid[key][0]
            filenames[i] = grid[key][1]
            del grid[key]

        self.coordinate_width = (abs(self.ul[0] - self.lr[0]) /
                                 (self.num_cols - 1))
        self.coordinate_height = (abs(self.ul[1] - self.lr[1])  /
                                  (self.num_rows - 1))

        row_w = w * self.num_cols
        row_h = h * self.num_rows
        scalar_w = float(self.window_width) / row_w
        scalar_h = float(self.window_height) / row_h
        resized_w = int(w * scalar_w)
        resized_h = int(h * scalar_h)

        row_w += self.num_cols
        row_h += self.num_rows

        for key in grid:
            grid[key] = grid[key].resize((resized_w, resized_h))
        rows = mandolin([grid[key] for key in sorted(grid.keys())],
                        self.num_cols)

        # Open pyglet window to allow users to select tiles
        window = pyglet.window.Window(self.window_width, self.window_height)
        cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
        window.set_mouse_cursor(cursor)
        window.set_caption('Select tiles to ignore')

        # Create pyglet sprite object
        batch = pyglet.graphics.Batch()
        sprites = []
        tiles = []
        n_row = 0  # index of row
        for row in rows:
            n_col = 0  # index of column
            for img in row:
                w, h = img.size
                x = n_col * (w + 1)
                y = self.window_height - (n_row + 1) * (h + 1)
                img = self.pil_to_pyglet(img, 'RGB')
                sprites.append(pyglet.sprite.Sprite(img, x=x, y=y, batch=batch))
                tiles.append((n_col, n_row))
                n_col += 1
            n_row += 1


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
            fp = os.path.join(self.source, 'points.apf')
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
            # Log indexes of skipped tiles
            fp = os.path.join(self.source, 'skipped.txt')
            print 'Writing {}...'.format(fp)
            indexes = []
            for tile in skip:
                n_col, n_row = tile
                i = n_col + self.num_cols * n_row
                indexes.append(i)
                # Move tiles into subfolder
                src = filenames[i]
                dst = os.path.join(self.source, 'skipped',
                                   os.path.basename(src))
                try:
                    shutil.move(src, dst)
                except:
                    os.mkdir(os.path.dirname(dst))
                    shutil.move(src, dst)
            with open(fp, 'wb') as f:
                f.write('\n'.join([str(i) for i in sorted(indexes)]))
            fp = os.path.join(self.source, 'selected.jpg')
            pyglet.image.get_buffer_manager().get_color_buffer().save(fp)
            window.close()
            pyglet.app.exit()
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
            time.sleep(0.1)

        pyglet.app.run()




    def get_center(self, coordinates):
        """Calculates center points for each title based on ul and lr"""
        x, y = coordinates
        center = (self.ul[0] + self.coordinate_width * x,
                  self.ul[1] - self.coordinate_height * y)
        return center




    def pil_to_pyglet(self, img, mode):
        """Convert PIL Image to pyglet image"""
        w, h = img.size
        raw = img.convert(mode).tobytes('raw', mode)
        return pyglet.image.ImageData(w, h, mode, raw, -w*len(mode))
