"""A collection of general use functions used by the stitch2d module"""

import os
import re
import shlex
import subprocess
import sys
from textwrap import fill




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
        print fill(s, subsequent_indent='  ')




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
    if isinstance(validator, (str, unicode)):
        validator = re.compile(validator, re.U)
    elif isinstance(validator, dict):
        prompt = '{}({}) '.format(prompt, '/'.join(validator.keys()))
    elif isinstance(validator, list):
        options = ['{}. {}'.format(x + 1, validator[x])
                   for x in xrange(0, len(validator))]
    else:
        raw_input(fill('Error in stitch2d.helpers.prompt: '
                       'Validator must be dict, list, or str.'))
        raise
    # Validate response
    loop = True
    while loop:
        # Print options
        if isinstance(validator, list):
            print '{}\n{}'.format('\n'.join(options), '-' * 60)
        # Prompt for value
        a = raw_input(prompt).decode(sys.stdin.encoding)
        if a.lower() == 'q':
            print 'User exited prompt'
            sys.exit()
        elif a.lower() == '?':
            print fill(helptext)
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
                result = unicode(result)
            except:
                result = str(result)
            loop = prompt('Is this value correct: "{}"?'.format(result),
                          {'y' : False, 'n' : True}, confirm=False)
        elif loop:
            print fill(errortext)
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
    mandolined = [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]
    remainder = len(lst) % n
    if remainder:
        leftovers = lst[-remainder:]
        mandolined.append(leftovers + [''] * (n - len(leftovers)))
    return mandolined
