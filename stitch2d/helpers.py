"""A collection of general use functions used by the stitch2d module"""
import os
import re
import shlex
import subprocess
import sys
from textwrap import fill

import numpy as np
from PIL import Image, ImageFilter
from PySide2.QtWidgets import QApplication, QFileDialog




IMAGE_MAP = {
    '.jpg' : 'JPEG',
    '.tif' : 'TIFF',
    '.tiff' : 'TIFF'
}

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




def cluster(data, maxgap):
    '''Group data such that successive elements differ by no more than maxgap

       Based on http://stackoverflow.com/questions/14783947

       Args:
           data (list): list of numbers (either floats or integers)
           maxgap (int): maximum acceptable gap between successive elements

       Returns:
           List of clusters
    '''
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def cprint(s, show=True):
    """Prints string only if conditional is true

    Args:
        s (str): string to print
        show (bool): specifies whether to print

    Returns:
        None
    """
    if bool(s) and show:
        print(fill(s, subsequent_indent='  '))


def prompt(prompt, validator, confirm=False,
           helptext='No help text provided', errortext='Invalid response!'):
    """Prompts user and validates response based on validator

    Args:
        prompt (str or unicode): prompt to display to user
        validator (str, list, or dict): object used to validate user
            response
        confirm (bool): specifies whether to have user verify response
        helptext (str or unicode): text to display if user enters '?'
        errortext (str or unicode): text to display if response
            does not validate

    Returns:
        A unicode string containing the validated user input

    Raises:
        Unspecified error: Validator is not dict, list, or str
    """
    # Prepare string
    prompt = u'{} '.format(prompt.rstrip())
    # Prepare validator
    if isinstance(validator, (str, str)):
        validator = re.compile(validator, re.U)
    elif isinstance(validator, dict):
        prompt = '{}({}) '.format(prompt, '/'.join(list(validator.keys())))
    elif isinstance(validator, list):
        options = ['{}. {}'.format(x + 1, validator[x])
                   for x in range(0, len(validator))]
    else:
        input(fill('Error in stitch2d.helpers.prompt: '
                       'Validator must be dict, list, or str.'))
        raise
    # Validate response
    loop = True
    while loop:
        # Print options
        if isinstance(validator, list):
            print('{}\n{}'.format('\n'.join(options), '-' * 60))
        # Prompt for value
        a = input(prompt)#.decode(sys.stdin.encoding)
        if a.lower() == 'q':
            print('User exited prompt')
            sys.exit()
        elif a.lower() == '?':
            print(fill(helptext))
            loop = False
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
                result = str(result)
            except:
                result = str(result)
            loop = prompt('Is this value correct: "{}"?'.format(result),
                          {'y' : False, 'n' : True}, confirm=False)
        elif loop:
            print(fill(errortext))
    # Return value as unicode
    return result


def mogrify(path, ext):
    """Uses ImageMagick to copy source files to a working directory

    Requires ImageMagick to be installed and on the system path.

    Args:
        path (str): filepath to directory containing images
        ext (str): extension of files to copy from path

    Returns:
        True if mogrify command succeeds, False if not
    """
    cprint('There was a problem opening some of the tiles!\n'
           'Copying tiles into a usable format...')
    ext = ext.strip('*.')
    subdir = os.path.join(path, 'working')
    try:
        os.mkdir(subdir)
    except OSError:
        pass
    cmd = 'mogrify -path "working" -format {0} *.{0}'.format(ext)
    args = shlex.split(cmd)
    try:
        subprocess.call(args, cwd=path)
    except:
        return False
    else:
        return True


def mandolin(lst, n):
    """Split list into groups of n members

    Based on http://stackoverflow.com/questions/9671224/

    Args:
        lst (list): list containing anything you like
        n (int): length of members

    Returns:
        List of lists of n members. The last value is padded
        with empty strings to n if the original list is not
        exactly divisible by n.
    """
    mandolined = [lst[i * n:(i + 1) * n] for i in range(len(lst) // n)]
    remainder = len(lst) % n
    if remainder:
        leftovers = lst[-remainder:]
        mandolined.append(leftovers + [''] * (n - len(leftovers)))
    return mandolined


def read_image(fp, mode=None):
    """Reads image using PIL"""
    im = Image.open(fp)
    # Workaround to handle 16-bit images
    data = np.array(im)
    maxval = np.max(data)
    if maxval > 255:
        # Calculate pixel values based on the range of values in the original
        def stretch(val, minval, maxval):
            return 255 * (val - minval) / (maxval - minval)
        minval = np.min(data)
        data = np.array([stretch(x, minval, maxval) for x in data])
        im = Image.fromarray(data)
    if mode:
        im = im.convert(mode)
    return im


def get_color(im):
    """Determines the base color of a simple image"""
    colors = [tuple([255 if ch else 0 for ch in color])
              for color in [t[1] for t in im.getcolors(512)]
              if not len(set(color)) == 1]
    if len(set(colors)) > 1:
        raise Exception('Color error: {}'.format(set(colors)))
    return colors[0]


def recolor(im, to_color):
    """Converts a simple image to a new color"""
    data = np.array(im).astype(np.uint16)
    to_color = COLORS.get(to_color, to_color)
    from_color = get_color(im)
    if to_color != from_color:
        from_channel = [i for i, val in enumerate(from_color) if val][0]
        for i, ch in enumerate(to_color):
            if ch and i != from_channel:
                data[...,i] = data[...,from_channel]
        # Zero any channel not found in to_color
        for i, ch in enumerate(from_color):
            if ch and not to_color[i]:
                data[...,i] = 0
    return Image.fromarray(data)


def brighten(im, minval):
    """Brightens an image using a minimum allowable value

    Args:
        im (str): path to image file
        minval (int): minimum channel value


    Returns:
       Brightened PIL Image
    """

    def func(val, minval):
        return minval + (255 - minval) * val // 255

    arr = np.array(im)
    arr[arr > 0] = np.apply_along_axis(func, 0, arr[arr > 0], minval=minval)
    return Image.fromarray(arr)


def blur(im, radius):
    """Blurs an image"""
    blur = ImageFilter.GaussianBlur(radius)
    try:
        return im.filter(blur)
    except ValueError:
        return im.convert('RGB').filter(blur)


def _select_folder(title='Select tileset directory'):
    """Select directory using GUI

    Args:
        title (str): title of GUI window

    Returns:
        Path as to directory as string
    """
    app = QApplication()
    return QFileDialog.getExistingDirectory(caption=title)


def _guess_extension(path):
    """Determines extension based on files in path

    Args:
        path (str): path to folder containing tiles

    Returns:
        File extension of first valid file type
    """
    for fn in os.listdir(path):
        ext = os.path.splitext(fn)[1]
        try:
            IMAGE_MAP[ext.lower()]
        except KeyError:
            pass
        else:
            return ext
    else:
        msg = (u'Could not find a valid tileset in {} Supported image'
                ' formats include {}').format(path, sorted(IMAGE_MAP))
        raise Exception(msg)


def _get_coordinates(fn):
    """Extracts coordinates from SEM filename

    Args:
       fn (str): path to an SEM tile

    Returns:
       Coordinates as a tuple
    """
    return tuple([int(c) for c in fn.split('@')[1].split(']')[0].split(' ')])
