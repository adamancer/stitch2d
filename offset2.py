#graphical interface to determine offset

import ctypes
import os
import pyglet
import random
import subprocess
from random import randint

import pyglet.text
import pyglet.window
from pyglet.window import key, mouse
from pyglet.gl import *

from PIL import Image

# 1. Get screen dimensions
# 2. Select tiles from near center of mosaic
# 3. Crop tiles to dimensions of window n(w=50%, h=100%)
# 4.

class Offset(object):

    def __init__(self, rows):
        """
        @param list  tiles must be ordered and patched!
        """
        # Tileset parameters
        self.rows = rows
        self.num_cols = len(rows[0])
        self.num_rows = len(rows)
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
        self.x_offset_within_row = 0  # final value should be <= 0
        self.x_offset_between_rows = 0
        self.y_offset_within_row = 0
        self.y_offset_between_rows = 0  # final value should be <= 0
        # Get window dimensions. These will be used to set the size of
        # the pyglet window later.
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        self.window_width = screen.width - 200
        self.window_height = screen.height - 200

        while True:
            self.get_tiles()
            raw_input()




    def get_tiles(self, from_middle=False, from_row=True):
        """Returns two adjacent tiles"""
        if from_middle:
            n_col = self.num_cols / 2
            n_row = self.num_rows / 2
        else:
            n_col = randint(0, self.num_cols - 1)
            n_row = randint(0, self.num_rows - 1)
        if from_row and not self.num_cols == 1:
            row = self.rows[n_row]
            try:
                tiles = row[n_col], row[n_col+1]
            except IndexError:
                tiles = row[n_col-1], row[n_col]
        elif not from_row and not self.num_rows == 1:
            row = self.rows[n_row]
        self.composite(tiles, from_row)



    def composite(self, tiles, from_row=True):
        left, right = [Image.open(tile).convert('RGBA') for tile in tiles]
        w, h = left.size
        crop_w = self.window_width / 2
        if crop_w > w:
            crop_w = w
        crop_h = self.window_height
        if crop_h > h:
            crop_h = h
        left = left.crop((w - crop_w, 0, w, crop_h))
        right = right.crop((0, 0, crop_w, crop_h))
        mask = Image.new('RGBA', (crop_w, crop_h), (255,255,255, 128))
        right = Image.alpha_composite(right, mask)
        #img = Image.new('RGB', (self.window_width, self.window_height))
        #img.paste(left, (0,0))
        #img.paste(right, (crop_w + self.x_offset_within_row, 0))

        # Open pyglet window to allow users to select tiles
        window = pyglet.window.Window(self.window_width, self.window_height)
        cursor = window.get_system_mouse_cursor(window.CURSOR_HAND)
        window.set_mouse_cursor(cursor)
        window.set_caption('Set offset between tiles in the same row')

        batch = pyglet.graphics.Batch()
        sprites = []
        img = self.pil_to_pyglet(left, 'RGBA')
        sprites.append(pyglet.sprite.Sprite(img, x=0, y=0, batch=batch))
        img = self.pil_to_pyglet(right, 'RGBA')
        x = crop_w + self.x_offset_within_row
        sprites.append(pyglet.sprite.Sprite(img, x=x, y=0, batch=batch))


        @window.event
        def on_draw():
            window.clear()
            batch.draw()


        @window.event
        def on_mouse_press(x, y, button, modifiers):
            window.clear()
            batch.draw()

        pyglet.app.run()




    def pil_to_pyglet(self, img, mode):
        """Convert PIL Image to pyglet image"""
        w, h = img.size
        raw = img.convert(mode).tobytes('raw', mode)
        return pyglet.image.ImageData(w, h, mode, raw, -w*len(mode))









'''
#define functions
def enq(s):
    #adds quotes to string
    return '"' + s + '"'

def get_image(path, rows, dim, ext,
              offset,
              win_w, win_h,
              this_row = -1, this_col = -1,
              same_row = True, blend = False):
    #get sequence parameters
    num_row = len(rows) #number of rows
    len_row = len(rows[0]) #length of individual rows
    #index for for if not set
    if this_row < 0:
        n = num_row / 2
    else:
        n = this_row
    #index for col if not set
    if len_row == 2:
        i = 0
    elif this_col < 0:
        i = len_row / 2
    else:
        i = this_col
    #get image parameters
    w = int(dim.split('x')[0]) #image width
    h = int(dim.split('x')[1]) #image height
    x = int(offset.split('x')[0]) #equivalent to x_row/x_col
    y = int(offset.split('x')[1]) #equivalent to y_row/y_col
    #adjacent images in same row (x2)
    if same_row:
        #print 'Processing within-row offset'
        #change signs
        x *= -1
        #get parameters for shave
        if w * 2 > win_w:
            dw = (w * 2 - win_w) / 2
        else:
            dw = 0
        if h > win_h:
            dh = (h - win_h) / 2
        else:
            dh = 0
        shave_this = str(dw) + 'x' + str(dh)
        #get next image in sequence
        j = i + 1
        #print rows[n][i]
        #print rows[n][j]
        img = [os.path.join(path, rows[n][i] + ext),
               os.path.join(rows[n][j] + ext)]
        #dim = str(w * j - x * (j - 1)) +\
        #      'x' +\
        #     str(h - y * (j - 1))
        dim = str(w * 2 - x) +\
              'x' +\
              str(h + y)
        cmd = ['convert',
               enq(img[0]),
               '-quiet',
               '-gravity',
               'SouthWest',
               '-extent',
               dim,
               enq(img[1]),
               '-gravity',
               'NorthEast',
               '-composite',
               '-gravity',
               'center',
               '-shave',
               shave_this,
               enq(os.path.join(path, 'temp.jpg'))]
        if blend:
            #add blend command
            cmd = cmd[0:10] +\
                  ['-compose', 'blend',
                   '-define', 'compose:args=50'] +\
                  cmd[10:]
        #print ' '.join(cmd)
        subprocess.call(' '.join(cmd),
                        #startupinfo=startupinfo,
                        shell=True)
    #adjacent images in same column (x2)
    else:
        #print 'Processing between-row offset'
        #change signs
        x *= -1
        #y *= -1
        #get parameters for shave
        if w > win_w:
            dw = (w - win_w) / 2
        else:
            dw = 0
        if h * 2 > win_h:
            dh = (h * 2 - win_h) / 2
        else:
            dh = 0
        shave_this = str(dw) + 'x' + str(dh)
        #get next image in seqeunce
        m = n + 1
        img = [os.path.join(path, rows[n][i] + ext),
               os.path.join(rows[m][i] + ext)]
        dim = str(w - x) +\
              'x' +\
              str(h * 2 - y)
        #print dim
        cmd = ['convert',
               enq(img[0]),
               '-quiet',
               '-extent',
               dim,
               enq(img[1]),
               '-gravity',
               'SouthEast',
               '-composite',
               '-gravity',
               'center',
               '-shave',
               shave_this,
               enq(os.path.join(path, 'temp.jpg'))]
        if blend:
            #add blend command
            cmd = cmd[0:8] +\
                  ['-compose', 'blend',
                   '-define', 'compose:args=50'] +\
                  cmd[8:]
        #print ' '.join(cmd)
        subprocess.call(' '.join(cmd),
                        #startupinfo=startupinfo,
                        shell=True)


def get_offset(path, rows, dim, ext, same_row):
    #set function variables to dict
    fx = dict()
    fx['last'] = None #coordinates of last click
    fx['path'] = path
    fx['rows'] = rows
    fx['ext'] = ext
    fx['dim'] = dim
    fx['offset'] = '0x0'
    fx['instructions'] = 'Click any distinct feature that appears ' +\
                         'on boths sides of the boundary.'
    #grid parameters
    fx['num_row'] = len(rows) - 2
    fx['len_row'] = len(rows[0]) - 2
    fx['n'] = -1 #row index
    fx['i'] = -1 #tile index
    fx['same_row'] = same_row
    #window parameters
    user32 = ctypes.windll.user32
    fx['w'] = user32.GetSystemMetrics(0) - 100
    fx['h'] = user32.GetSystemMetrics(1) - 100
    #cap window size to max_dim
    max_dim = 800
    if fx['w'] * fx['h'] > max_dim**2:
        if fx['w'] > fx['h']:
            fx['h'] = fx['h'] * max_dim / fx['w']
            fx['w'] = max_dim
        else:
            fx['w'] = fx['w'] * max_dim / fx['h']
            fx['h'] = max_dim
    #create initial image
    get_image(fx['path'], fx['rows'], fx['dim'], fx['ext'],
              '0x0',
              fx['w'], fx['h'],
              fx['n'], fx['i'],
              fx['same_row'])

    #create fullscreen window
    window = pyglet.window.Window(fx['w'], fx['h'])
    if same_row:
        window.set_caption('Set offset between adjacent ' +\
                           'tiles in the same row')
    else:
        window.set_caption('Set offset between adjacent ' +\
                           'tiles in the same column')
    #window.set_fullscreen(True, None)

    #load comparison image
    pic = pyglet.image.load(os.path.join(path, 'temp.jpg'))
    batch = pyglet.graphics.Batch()
    background = pyglet.graphics.OrderedGroup(0)
    fx['bg'] = pyglet.sprite.Sprite(pic, batch=batch,
                                    group=background)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        last_click = fx['last']
        #process coordinates
        if x <= 32 and y <= 32:
            #BOTTOM LEFT
            #accept offset and close
            window.close()
        elif x <= 32 and y >= window.height - 32:
            #TOP LEFT
            #show blend based on current offset
            temp = get_image(fx['path'], fx['rows'], fx['dim'], fx['ext'],
                             fx['offset'],
                             fx['w'], fx['h'],
                             fx['n'], fx['i'],
                             fx['same_row'], True)
            fx['bg'].image = pyglet.image.load(os.path.join(path, 'temp.jpg'))
        elif x >= window.width - 32 and y >= window.height - 32:
            #TOP RIGHT
            #get new tiles, never from edge
            fx['n'] = random.randint(1, fx['num_row'])
            fx['i'] = random.randint(1, fx['len_row'])
            temp = get_image(fx['path'], fx['rows'], fx['dim'], fx['ext'],
                             fx['offset'],
                             fx['w'], fx['h'],
                             fx['n'], fx['i'],
                             fx['same_row'], False)
            fx['bg'].image = pyglet.image.load(os.path.join(path, 'temp.jpg'))
        elif x >= window.width - 32 and y <= 32:
            #BOTTOM RIGHT
            #reset to default parameters
            fx['offset'] = '0x0'
            get_image(fx['path'], fx['rows'], fx['dim'], fx['ext'],
                      '0x0',
                      fx['w'], fx['h'],
                      fx['n'], fx['i'],
                      fx['same_row'])
            fx['bg'].image = pyglet.image.load(os.path.join(path, 'temp.jpg'))
        elif last_click:
            this_click = (x, y)
            #calculate offset based on this_click and last_click
            dx = last_click[0] - this_click[0]
            dy = last_click[1] - this_click[1]
            if fx['same_row'] and dx > 0:
                #second click to right of first
                dx *= -1
                dy *= -1
            elif fx['same_row']:
                #second click to left of first
                pass
            elif not fx['same_row'] and dy > 0:
                #second click above first
                pass
            elif not fx['same_row']:
                #second click below first
                dx *= -1
                dy *= -1
            #update labels
            fx['instructions'] = 'Click any distinct feature that appears ' +\
                                 'on boths sides of the boundary.'
            fx['offset'] = str(dx) + 'x' + str(dy)
            #reset last_click to None
            fx['last'] = None
        else:
            #update labels
            fx['instructions'] = 'Now click the same feature on the opposite ' +\
                                 'side of the boundary'
            #set last_click to current coordinates
            fx['last'] = (x, y)



    @window.event
    def on_key_press(symbol, modifiers):
        offset = fx['offset'].split('x')
        if symbol == key.LEFT:
            offset[0] = str(int(offset[0]) - 1)
        elif symbol == key.RIGHT:
            offset[0] = str(int(offset[0]) + 1)
        elif symbol == key.DOWN and fx['same_row']:
            offset[1] = str(int(offset[1]) - 1)
        elif symbol == key.UP and fx['same_row']:
            offset[1] = str(int(offset[1]) + 1)
        #invert up/down for column offset
        elif symbol == key.DOWN:
            offset[1] = str(int(offset[1]) + 1)
        elif symbol == key.UP:
            offset[1] = str(int(offset[1]) - 1)
        fx['offset'] = offset[0] + 'x' + offset[1]
        temp = get_image(fx['path'], fx['rows'], fx['dim'], fx['ext'],
                         fx['offset'],
                         fx['w'], fx['h'],
                         fx['n'], fx['i'],
                         fx['same_row'], True)
        fx['bg'].image = pyglet.image.load(os.path.join(path, 'temp.jpg'))


    #prepare labels
    guide = pyglet.text.Label(fx['instructions'],
                              font_name = 'Verdana', font_size = 10,
                              bold = True, color = (0,255,255,255),
                              x = window.width / 2, y = window.height - 2,
                              anchor_x = 'center', anchor_y = 'top')
    offset = pyglet.text.Label(fx['offset'],
                               font_name = 'Verdana', font_size = 10,
                               bold = True, color = (0,255,255,255),
                               x = window.width / 2, y = 2,
                               anchor_x = 'center', anchor_y = 'bottom')
    #corners
    top_l = pyglet.text.Label('show',
                              font_name = 'Verdana', font_size = 10,
                              bold = True, color = (0,255,255,255),
                              x = 2, y = window.height - 2,
                              anchor_x = 'left', anchor_y = 'top')
    top_r = pyglet.text.Label('new',
                              font_name = 'Verdana', font_size = 10,
                              bold = True, color = (0,255,255,255),
                              x = window.width - 2, y = window.height - 2,
                              anchor_x = 'right', anchor_y = 'top')
    bot_r = pyglet.text.Label('reset',
                              font_name = 'Verdana', font_size = 10,
                              bold = True, color = (0,255,255,255),
                              x = window.width - 2, y = 2,
                              anchor_x = 'right', anchor_y = 'bottom')
    bot_l = pyglet.text.Label('done',
                              font_name = 'Verdana', font_size = 10,
                              bold = True, color = (0,255,255,255),
                              x = 2, y = 2,
                              anchor_x = 'left', anchor_y = 'bottom')


    #draw screen
    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        #update labels
        guide.text = fx['instructions']
        guide.draw()
        offset.text = fx['offset']
        offset.draw()
        top_l.draw()
        top_r.draw()
        bot_r.draw()
        bot_l.draw()

    #run the application
    pyglet.app.run()

    #return accepted value on window close
    return fx['offset']
'''
