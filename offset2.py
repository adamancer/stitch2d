#graphical interface to determine offset

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
from pyglet.window import key, mouse
from pyglet.gl import *

from PIL import Image




class OffsetEngine(pyglet.window.Window):


    def __init__(self, rows, offsets=None, *args, **kwargs):
        super(OffsetEngine, self).__init__(*args, **kwargs)
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
        if offsets:
            self.x_offset_within_row = offsets[0]
            self.y_offset_within_row = offsets[1]
        self.coordinates = []

        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_default_screen()
        self.width = screen.width - 200
        self.height = screen.height - 200

        self.set_caption('Set offset between tiles in the same row')
        self.hand = self.get_system_mouse_cursor(self.CURSOR_HAND)
        self.crosshair = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(self.crosshair)

        self.guidance = [
            ('Click any distinct feature that appears on boths sides'
             ' of the boundary'),
            ('Click the corresponding feature on the other side'
             ' of the boundary'),
            ('Use the arrow keys to adjust the offset or click reset'
             ' to start over')
              ]
        self.orig_guidance = copy(self.guidance)

        self.label_batch = pyglet.graphics.Batch()
        self.labels = {}
        self.labels['guidance'] = pyglet.text.Label(
            self.guidance.pop(0),
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
            'Reset offset to 0x0',
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
        for key in self.labels:
            self.labels[key].bold = True

        self.get_tiles()
        pyglet.app.run()




    def on_draw(self):
        self.clear()
        self.image_batch.draw()
        self.label_batch.draw()




    def on_close(self):
        print 'Closing...'
        #return (self.x_offset_within_row, self.y_offset_within_row,
        #        self.x_offset_between_rows, self.y_offset_between_rows)




    def on_mouse_motion(self, x, y, dx, dy):
        for key in ['new', 'reset', 'save']:
            label = self.labels[key]
            x1, y1, x2, y2 = self.calculate_label_position(label)
            if x1 < x < x2 and y1 < y < y2:
                label.color = (255,0,0,255)
                self.set_mouse_cursor(self.hand)
                break
            else:
                label.color = (255,255,255,255)
        else:
            self.set_mouse_cursor(self.crosshair)
        time.sleep(0.1)




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
                    if True:
                        self.x_offset_within_row = 0
                        self.y_offset_within_row = 0
                    else:
                        self.x_offset_between_rows = 0
                        self.y_offset_between_rows = 0
                else:
                    self.close()
                break
        # This conditional handles calculating the actual offset
        # once the user has made the first two clicks
        else:
            if len(self.coordinates) < 2:
                self.coordinates.append((x, y))
                self.labels['guidance'].text = self.guidance.pop(0)
                if len(self.coordinates) == 2:
                    self.coordinates.sort(key=lambda s:s[0])
                    x1y1, x2y2 = self.coordinates
                    x1, y1 = x1y1
                    x2, y2 = x2y2
                    self.x_offset_within_row = x1 - x2
                    self.y_offset_within_row = y2 - y1
                    self.labels['offset'].text = '{}x{}'.format(
                        self.x_offset_within_row,
                        self.y_offset_within_row
                    )




    def on_key_press(self, symbol, modifiers):
        """Handles key presses for fine-tuning the offset"""
        if symbol == key.LEFT:
            pass




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
        crop_w = self.width / 2
        if crop_w > w:
            crop_w = w
        crop_h = self.height
        if crop_h > h:
            crop_h = h
        left = left.crop((w - crop_w, 0, w, crop_h))
        right = right.crop((0, 0, crop_w, crop_h))
        mask = Image.new('RGBA', (crop_w, crop_h), (255,255,255,128))
        right = Image.alpha_composite(right, mask)

        # Adding sprites to a batch is insufficient to keep them
        # from being trashed. They should be saved in a class variable
        # instead.
        self.sprites = []
        self.image_batch = pyglet.graphics.Batch()
        img = self.pil_to_pyglet(left, 'RGBA')
        self.sprites.append(pyglet.sprite.Sprite(img, x=0, y=0,
                                                 batch=self.image_batch))
        img = self.pil_to_pyglet(right, 'RGBA')
        x = crop_w + self.x_offset_within_row
        self.sprites.append(pyglet.sprite.Sprite(img, x=x, y=0,
                                                 batch=self.image_batch))




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
