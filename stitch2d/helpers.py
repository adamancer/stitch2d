import os
import re
import shlex
import subprocess
import sys
from textwrap import fill




def cprint(s, show=True):
    """Conditional print"""
    if bool(s) and show:
        print fill(s, subsequent_indent='  ')




def prompt(prompt, validator, confirm=False,
           helptext='No help text provided', errortext='Invalid response!'):
    """Prompts user and validates response based on validator

    @param string
    @param regex, list, or dict
    @param boolean
    @param string
    @param string
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
    """Saves copy of tiles to subfolder"""
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

    @param list
    @param int
    @return list
    """
    mandolined = [lst[i*n:(i+1)*n] for i in range(len(lst) / n)]
    remainder = len(lst) % n
    if remainder:
        leftovers = lst[-remainder:]
        mandolined.append(leftovers + [''] * (n - len(leftovers)))
    return mandolined
