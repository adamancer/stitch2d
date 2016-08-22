import glob
import os

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .organizer import _get_name


COLORS = {
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'cyan': (0, 255, 255),
    'magenta':(255, 0, 255),
    'yellow': (255, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

def convert(from_color, to_color):
    # Convert to_color to rgb if given as string
    try:
        to_color = COLORS[to_color.lower()]
    except KeyError:
        pass
    # Channels in the from_color must covary
    if len(set(from_color)) == 3:
        return
    try:
        val = [ch for ch in from_color if ch].pop()
    except IndexError:
        val = 0
    rgb = tuple([val if ch else 0 for ch in to_color])
    return rgb


def rainbow():
    src = 'blue'
    im = Image.open('{}.tif'.format(src))
    im = im.convert('RGB')
    colors = [t[1] for t in im.getcolors()]

    data = np.array(im)
    red, green, blue = data.T

    for color in COLORS:
        _data = data.copy()
        for r, g, b in set(colors):
            pixels = (red == r) & (blue == b) & (green == g)
            _data[...,:][pixels.T] = convert((r, g, b), color)
        Image.fromarray(_data).save('_{}.tif'.format(color))

    src = 'green'
    im = Image.open('{}.tif'.format(src))
    im = im.convert('RGB')

    src = 'blue'
    _im = Image.open('_{}.tif'.format(src))
    _im.convert('RGB')

    print list(_im.getdata()) == list(im.getdata())


def normalize():
    im = Image.open(fp).convert('RGB')
    data = np.array(im)
    data = data.astype(np.uint16)
    red, green, blue = data.T
    colors = [t[1] for t in im.getcolors()]
    # Determine color
    for color in [color for color in colors if set(color) > 1]:
        mask = tuple([255 if ch else 0 for color in colors])
        for key, val in COLORS.iteritems():
            if mask == val:
                color = key
                break


def composite(path, colormap):
    """Create a composite using a set of images and colors

    Args:
        path (str): path to images
        colormap (dict): maps colors to elements
    """
    ordered = ('red', 'green', 'blue', 'cyan', 'magenta', 'yellow')
    colors = [color for color in ordered if color in colormap]
    colors = '-'.join(colors)
    print 'Making {} composite from images in {}...'.format(colors, path)
    # Path to font
    ttf = os.path.join(os.path.dirname(__file__), 'files',
                       'OpenSans-Regular.ttf')
    # Map images in source folder to the appropriate element
    elementmap = {}
    for fp in glob.glob(os.path.join(path, '*.tif')):
        fn = os.path.basename(fp)
        try:
            element = _get_name(fn).split('_').pop()
        except AttributeError:
            try:
                element = os.path.splitext(fn)[0].split('_').pop()
            except IndexError:
                continue
        elementmap[element] = fp
    # Recolor the images
    legend = []
    for color in colormap:
        element = colormap[color]
        fp = elementmap[element]
        im = Image.open(fp).convert('RGB')
        data = np.array(im)
        data = data.astype(np.uint16)
        red, green, blue = data.T
        colors = [t[1] for t in im.getcolors(512)]
        # Ignore grays when recoloring if color is not white
        if color != 'white':
            colors = [_color for _color in colors if len(set(_color)) > 1]
        for r, g, b in set(colors):
            pixels = (red == r) & (blue == b) & (green == g)
            data[...,:][pixels.T] = convert((r, g, b), color)
        try:
            composite = np.add(composite, data)
        except NameError:
            composite = data
        # Add file to legend
        legend.append((element, color))
    legend.sort(key=lambda row:ordered.index(row[1]))
    w, h = im.size
    # Set pixel value to 255
    composite[composite > 255] = 255
    composite = Image.fromarray(composite.astype(np.uint8))
    # Add space for legend at bottom of mosaic
    label_height = int(im.size[1] * 0.04)
    legend_height = int(label_height * (len(legend) - 1))
    im = Image.new('RGB', (im.size[0], im.size[1] + legend_height), 'black')
    im.paste(composite, (0, 0))
    # Draw legend
    draw = ImageDraw.Draw(im)
    x1, y1 = (0, im.size[1])
    x2, y2 = (im.size[0], im.size[1] - legend_height - label_height)
    draw.rectangle(((x1, y1), (x2, y2)), fill='black')
    for i in xrange(len(legend)):
        element, color = legend[::-1][i]
        # Add a color square
        x = int(0.95 * im.size[0])
        y = im.size[1] - (i + 1) * int(label_height)
        w, h = [dim * 0.04 for dim in im.size]
        draw.rectangle(((x, y), (x + w, y + h)), fill=color)
        # Add the label
        x -= int(w * 1.5)
        size = 100
        font = ImageFont.truetype(ttf, size)
        w, h = font.getsize(element)
        size = int(0.8 * size * label_height / float(h))
        font = ImageFont.truetype(ttf, size)
        draw.text((x, y), element, (255, 255, 255), font=font)
    # Draw label
    base = os.path.dirname(fp)
    text = '{} (multielement X-ray map)'.format(base)
    x = int(0.02 * im.size[0])
    y = im.size[1] - legend_height - label_height
    draw.text((x, y), text, 'white', font=font)
    # Set filename
    elements = [colormap[color] for color in ordered if color in colormap]
    print ' Saving TIFF...'
    im.save('{}_{}.tif'.format(base, ''.join(elements)))
    print ' Saving JPEG 2000...'
    im.save('{}_{}.jp2'.format(base, ''.join(elements)))