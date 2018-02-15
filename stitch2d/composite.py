import glob
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

from .helpers import _guess_extension, _select_folder
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



def brighten(val, minval):
    """Brightens image based on minval

    Args:
        val (int): value of a single channel
        minval (int): minimum channel value in the converted image. Used to
            brighten or darken the composite.

    Returns:
        New channel value as int
    """
    return minval + (255 - minval) * val / 255




def convert(from_color, to_color, minval=None):
    """Converts color name to rgb

    Args:
        from_color (str, tuple): the name or RGB representation of a color.
            The color must be one of red, green, blue, cyan, magenta, yellow,
            black, or white.
        to_color (str, tuple): the name or RGB representation of a color
        minval (int): minimum channel value in the converted image. Used to
            brighten or darken the composite.

    Returns:
        Tuple containing the converted color as RGB
    """
    try:
        to_color = COLORS[to_color.lower()]
    except KeyError:
        pass
    # The source color must be a tint/shade of one of the eight colors
    if len(set(from_color)) == 3:
        raise Exception('Color must be one of {}'.format(sorted(colors)))
    try:
        val = [ch for ch in from_color if ch].pop()
    except IndexError:
        val = 0
    # Adjust the intensity of the color based on minval
    if minval is not None:
        if not 0 <= minval <= 255:
            raise Exception('minval must be between 0 and 255')
        val = minval + (255 - minval) * val / 255
    rgb = tuple([int(val) if ch else 0 for ch in to_color])
    return rgb




def composite(path=None, output='.', label=None, jpeg=False, minval=None,
              blur=0, **colormap):
    """Creates a composite element map

    Args:
        path (str): path to set of images to be composited
        label (str): label for composite image
        jpeg (bool): specifies whether to create a JPEG 2000 derivative
        **colormap: keyword arguments of the form color=element
    """
    if path is None:
        path = _select_folder()
    ordered = ('red', 'green', 'blue', 'cyan', 'magenta', 'yellow')
    colors = [color for color in ordered if color in colormap]
    colors = '-'.join(colors)
    print 'Making {} composite from images in {}...'.format(colors, path)
    # Path to font
    ttf = os.path.join(os.path.dirname(__file__), 'files',
                       'OpenSans-Regular.ttf')
    # Map images in source folder to the appropriate element
    ext = _guess_extension(path)
    elementmap = {}
    for fp in glob.glob(os.path.join(path, '*' + ext)):
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
    for to_color in colormap:
        element = colormap[to_color]
        print ' Processing {}...'.format(element)
        fp = elementmap[element]
        im = Image.open(fp).convert('RGB')
        data = np.array(im)
        data = data.astype(np.uint16)
        # Identify from_color
        colors = [tuple([255 if ch else 0 for ch in color])
                  for color in [t[1] for t in im.getcolors(512)]
                  if not len(set(color)) == 1]
        if len(set(colors)) > 1:
            raise Exception('Color error: {}'.format(set(colors)))
        # Populate color channels based on intensities in from_color
        from_color = colors[0]
        if from_color != COLORS[to_color]:
            from_channel = [i for i, val in enumerate(from_color) if val][0]
            for i, ch in enumerate(COLORS[to_color]):
                if ch and i != from_channel:
                    data[...,i] = data[...,from_channel]
            # Zero any channel not found in to_color
            for i, ch in enumerate(from_color):
                if ch and not COLORS[to_color][i]:
                    data[...,i] = 0
        # Apply modified image to the composite
        try:
            composite = np.add(composite, data)
        except NameError:
            composite = data
        # Add file to legend
        legend.append((element, to_color))
    legend.sort(key=lambda row:ordered.index(row[1]))
    w, h = im.size
    # Normalize brightness if minval is specified
    composite[composite < 10] = 0
    if minval:
        print ' Brightening image...'
        composite[composite > 0] = np.apply_along_axis(brighten, 0,
                                                       composite[composite > 0],
                                                       minval=minval)
    # Set pixel value to 255
    composite[composite > 255] = 255
    composite = Image.fromarray(composite.astype(np.uint8))
    # Add space for legend at bottom of mosaic
    label_height = int(im.size[1] * 0.04)
    legend_height = int(label_height * (len(legend) - 1))
    im = Image.new('RGB', (im.size[0], im.size[1] + legend_height), 'black')
    im.paste(composite, (0, 0))
    if blur:
        print ' Blurring image...'
        im = im.filter(ImageFilter.GaussianBlur(radius=blur))
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
    label = os.path.basename(os.path.dirname(fp)) if label is None else label
    text = '{} (multielement X-ray map)'.format(label)
    x = int(0.02 * im.size[0])
    y = im.size[1] - legend_height - label_height
    draw.text((x, y), text, 'white', font=font)
    # Set filename
    elements = [colormap[color] for color in ordered if color in colormap]
    fn = label.replace(' ', '_')
    print ' Saving TIFF...'
    fp = os.path.join(output, '{}_{}.tif'.format(fn, ''.join(elements)))
    im.save(fp)
    if jpeg:
        print ' Saving JPEG 2000...'
        im.save(os.path.splitext(fp)[0] + '.jp2')
