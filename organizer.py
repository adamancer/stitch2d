import glob
import os
import re
import shutil


def get_root():
    return 'C:\\MinSci'


def get_name(f):
    specimen = re_specimen.match(os.path.splitext(f)[0]).group().capitalize()
    #clean specimen name
    if not '_' in specimen:
        specimen += '_'
    #check end of filename against keywords/patterns and return filename
    if re_map.search(os.path.splitext(f)[0]):
        element = re_map.search(os.path.splitext(f)[0]).group(1)
        return specimen + element.capitalize()
    elif 'RefGrey' in f:
        return specimen + 'nbsed'
    elif 'Grey' in f:
        return specimen + 'bsed'
    else:
        return None



#regular expressions to capture sample information
re_grid = re.compile('\[.+\]')
re_specimen = re.compile('[^\[]+')
re_map = re.compile('_([A-z]{1,2})_[A-z+]{1,3}_map')

#sort and move images
directories = []
for f in glob.iglob(os.path.join('*.tif')):
    #get directory name
    d = get_name(os.path.basename(f))
    if d:
        # Create directory if neccesary
        try:
            os.mkdir(os.path.join(get_root(), 'Workflows', 'Mosaics', d))
        except:
            pass
        else:
            print 'Creating {}'.format(d)
        # Move file into proper directory
        src = os.path.join(os.getcwd(), f)
        dst = os.path.join(get_root(), 'Workflows', 'Mosaics', d, f)
        try:
            open(dst, 'rb')
        except:
            print 'Copying {}...'.format(os.path.basename(dst))
            shutil.copy2(src, dst)
        else:
            #print '{} already exists!'.format(os.path.basename(dst))
            pass

#notify user that script is complete
raw_input('Done! Press any key to exit.')
