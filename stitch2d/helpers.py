import re
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
