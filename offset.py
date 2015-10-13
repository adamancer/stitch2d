import ctypes
import os
import pyglet
import random
import subprocess
import sys
import time
from copy import copy
from random import randint

import pyglet.text
import pyglet.window
from PIL import Image
from pyglet.window import key, mouse
from pyglet.gl.base import CanvasConfig, Context




class OffsetEngine(pyglet.window.Window):


    def __init__(self, rows, same_row=True, offsets=None, *args, **kwargs):
        # Set window size in init because Windows doesn't pick up
        # set_size() reliably.
        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        margin = 75
        super(OffsetEngine, self).__init__(
            *args,
            width=screen.width - margin * 2,
            height=screen.height - margin * 2,
            visible=False,
            **kwargs)
        self.set_location(margin, margin)

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
        self.coordinates = []
        if offsets:
            try:
                self.x_offset_within_row = offsets[0]
                self.x_offset_between_rows = 0
                self.y_offset_within_row = offsets[1]
                self.y_offset_between_rows = 1
            except IndexError:
                print 'Provided offsets no good'
        else:
            self.x_offset_within_row = 1  # final value should be <= 0
            self.x_offset_between_rows = 0
            self.y_offset_within_row = 0
            self.y_offset_between_rows = 0  # final value should be <= 0

        self.hand = self.get_system_mouse_cursor(self.CURSOR_HAND)
        self.crosshair = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(self.crosshair)

        self.guidance = [
            ('Click any distinct feature that appears on both sides'
             ' of the boundary, or try the arrow keys for minor'
             ' adjustsments'),
            ('Click the corresponding feature on the other side'
             ' of the boundary'),
            ('Use the arrow keys to adjust the offset or click reset'
             ' to start over')
              ]
        self.orig_guidance = copy(self.guidance)

        self.same_row = same_row
        if self.same_row:
            self.set_caption('Set offset between tiles in the same row')
        else:
            self.set_caption('Set offset between tiles in different rows')

        self.n_row = 0
        self.n_col = 0

        self.color = (255,255,255,255)
        self.label_batch = pyglet.graphics.Batch()
        self.labels = {}
        self.labels['guidance'] = pyglet.text.Label(
            self.guidance.pop(0),
            width = self.width / 2,
            align = 'center',
            multiline = True,
            x = self.width / 2,
            y = self.height - 8,
            anchor_x='center',
            anchor_y='top',
            batch=self.label_batch)
        self.labels['offset'] = pyglet.text.Label(
            '{}x{}'.format(self.x_offset_within_row, self.y_offset_within_row),
            x = self.width / 2,
            y = 8,
            anchor_x='center',
            anchor_y='bottom',
            batch=self.label_batch)
        self.labels['new'] = pyglet.text.Label(
            'Get different tiles',
            x = self.width - 8,
            y = self.height - 8,
            anchor_x='right',
            anchor_y='top',
            batch=self.label_batch)
        self.labels['reset'] = pyglet.text.Label(
            'Reset offset',
            x = 8,
            y = self.height - 8,
            anchor_x='left',
            anchor_y='top',
            batch=self.label_batch)
        self.labels['save'] = pyglet.text.Label(
            'Save and return',
            x = self.width - 8,
            y = 8,
            anchor_x='right',
            anchor_y='bottom',
            batch=self.label_batch)
        self.labels['coordinates'] = pyglet.text.Label(
            'Tile: {}x{}'.format(self.n_row, self.n_col),
            x = 8,
            y = 8,
            anchor_x='left',
            anchor_y='bottom',
            batch=self.label_batch)
        for key in self.labels:
            self.labels[key].bold = True
            self.labels[key].color = self.color

        self.get_tiles()
        self.set_visible(True)




    def set_offset(self):
        pyglet.app.run()
        return (self.x_offset_within_row, self.y_offset_within_row,
                self.x_offset_between_rows, self.y_offset_between_rows)




    def on_draw(self):
        pyglet.gl.glClearColor(0,0,0,255)
        self.clear()
        self.image_batch.draw()
        self.label_batch.draw()




    def on_mouse_motion(self, x, y, dx, dy):
        for key in ['new', 'reset', 'save']:
            label = self.labels[key]
            x1, y1, x2, y2 = self.calculate_label_position(label)
            if x1 < x < x2 and y1 < y < y2:
                label.color = (255,0,0,255)
                self.set_mouse_cursor(self.hand)
                break
            else:
                label.color = self.color
        else:
            self.set_mouse_cursor(self.crosshair)
        time.sleep(0.05)




    def on_mouse_press(self, x, y, button, modifiers):
        for key in ['new', 'reset', 'save']:
            label = self.labels[key]
            x1, y1, x2, y2 = self.calculate_label_position(label)
            if x1 < x < x2 and y1 < y < y2:
                if key == 'new':
                    self.get_tiles(from_middle=False)
                elif key == 'reset':
                    self.coordinates = []
                    self.guidance = copy(self.orig_guidance)
                    self.labels['guidance'].text = self.guidance.pop(0)
                    if self.same_row:
                        self.x_offset_within_row = 1
                        self.y_offset_within_row = 0
                    else:
                        self.x_offset_between_rows = 0
                        self.y_offset_between_rows = 1
                    self.apply_offset()
                elif key == 'save':
                    self.close()
                break
        # This conditional handles calculating the actual offset
        # once the user has made the first two clicks
        else:
            if len(self.coordinates) < 2:
                self.coordinates.append((x, y))
                self.labels['guidance'].text = self.guidance.pop(0)
                if len(self.coordinates) == 2:
                    if self.same_row:
                        self.coordinates.sort(key=lambda s:s[0])
                        x1y1, x2y2 = self.coordinates
                        x1, y1 = x1y1
                        x2, y2 = x2y2
                        self.x_offset_within_row = x1 - x2
                        self.y_offset_within_row = y2 - y1
                    else:
                        self.coordinates.sort(key=lambda s:s[1])
                        x1y1, x2y2 = self.coordinates
                        x1, y1 = x1y1
                        x2, y2 = x2y2
                        self.x_offset_between_rows = x2 - x1
                        self.y_offset_between_rows = y1 - y2
                    self.apply_offset()




    def on_key_press(self, symbol, modifiers):
        """Handles key presses for fine-tuning the offset"""
        if len(self.coordinates) < 2:
            self.coordinates = [(0,0),(0,1)]
            self.labels['guidance'].text = self.guidance.pop()
        if self.same_row:
            if symbol == key.LEFT:
                self.x_offset_within_row -= 1
            elif symbol == key.RIGHT:
                self.x_offset_within_row += 1
            elif symbol == key.UP:
                self.y_offset_within_row -= 1
            elif symbol == key.DOWN:
                self.y_offset_within_row += 1
        else:
            if symbol == key.LEFT:
                self.x_offset_between_rows -= 1
            elif symbol == key.RIGHT:
                self.x_offset_between_rows += 1
            elif symbol == key.UP:
                self.y_offset_between_rows -= 1
            elif symbol == key.DOWN:
                self.y_offset_between_rows += 1
        self.apply_offset()




    def get_tiles(self, from_middle=False):
        """Returns two adjacent tiles"""
        if from_middle:
            n_col = self.num_cols / 2
            n_row = self.num_rows / 2
        else:
            n_col = randint(0, self.num_cols - 1)
            n_row = randint(0, self.num_rows - 1)
        if self.same_row and self.num_cols > 1:
            row = self.rows[n_row]
            try:
                tiles = row[n_col], row[n_col+1]
            except IndexError:
                n_col -= 1
                tiles = row[n_col], row[n_col+1]
        elif not self.same_row and self.num_rows > 1:
            try:
                self.rows[n_row+1]
            except IndexError:
                n_row -= 1
            try:
                tiles = [self.rows[n_row][n_col], self.rows[n_row+1][n_col]]
            except:
                n_col -= 1
                tiles = [self.rows[n_row][n_col], self.rows[n_row+1][n_col]]
        self.n_row = n_row
        self.n_col = n_col
        self.labels['coordinates'].text = 'Tile: {}x{}'.format(n_row, n_col)
        # If selector has been run on the tileset, there will be
        # gaps represented by empty strings. From a content
        # standpoint, we don't care about these, but we need to
        # try them because PIL can't process them.
        try:
            tiles = [Image.open(tile).convert('RGBA') for tile in tiles]
        except:
            self.get_tiles()
        else:
            return self.composite(tiles)




    def composite(self, tiles):
        t1, t2 = tiles
        w, h = t1.size
        if self.same_row:
            crop_w = self.width / 2
            if crop_w > w:
                crop_w = w
            crop_h = self.height
            if crop_h > h:
                crop_h = h
            t1 = t1.crop((w - crop_w, 0, w, crop_h))  # left tile
            t2 = t2.crop((0, 0, crop_w, crop_h))      # right tile
            x1, y1 = 0, 0
            x2, y2 = crop_w, 0
        else:
            crop_w = self.width
            if crop_w > w:
                crop_w = w
            crop_h = self.height / 2
            if crop_h > h:
                crop_h = h
            t1 = t1.crop((0, h - crop_h, crop_w, h))  # top tile
            t2 = t2.crop((0, 0, crop_w, crop_h))       # bottom tile
            x1, y1 = 0, crop_h
            x2, y2 = 0, 0

        # Pyglet draws from 0,0 in the ___er left
        # Adding sprites to a batch is insufficient to keep them
        # from being trashed. They must be saved in a class variable.
        self.sprites = []
        self.image_batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)
        img = self.pil_to_pyglet(t1, 'RGBA')
        self.sprites.append(pyglet.sprite.Sprite(img, x=x1, y=y1,
                                                 batch=self.image_batch,
                                                 group=self.background))
        img = self.pil_to_pyglet(t2, 'RGBA')
        self.sprites.append(pyglet.sprite.Sprite(img, x=x2, y=y2,
                                                 batch=self.image_batch,
                                                 group=self.foreground))
        self.apply_offset()
        return self




    def apply_offset(self):
        w, h = 0, 0
        if self.same_row:
            w = self.width / 2
            x = self.x_offset_within_row
            y = self.y_offset_within_row
        else:
            h = self.height / 2
            x = self.x_offset_between_rows
            y = self.y_offset_between_rows
        self.sprites[1].x = w + x
        self.sprites[1].y = -y
        self.labels['offset'].text = '{}x{}'.format(x, y)
        if not self.labels['offset'].text in ('0x1','1x0'):
            self.sprites[1].opacity = 144
        else:
            self.sprites[1].opacity = 255
        return self




    def pil_to_pyglet(self, img, mode):
        """Convert PIL Image to pyglet image"""
        w, h = img.size
        raw = img.convert(mode).tobytes('raw', mode)
        return pyglet.image.ImageData(w, h, mode, raw, -w*len(mode))




    def calculate_label_position(self, label):
        if label.anchor_x == 'left':
            x1 = label.x
            x2 = label.x + label.content_width
        elif label.anchor_x == 'right':
            x1 = label.x - label.content_width
            x2 = label.x
        else:
            x1 = label.x - label.content_width / 2
            x2 = label.x + label.content_width / 2
        if label.anchor_y == 'top':
            y1 = label.y - label.content_height
            y2 = label.y
        elif label.anchor_y == 'bottom':
            y1 = label.y
            y2 = label.y + label.content_height
        else:
            y1 = label.y - label.content_height / 2
            y2 = label.y + label.content_height / 2
        return x1, y1, x2, y2
