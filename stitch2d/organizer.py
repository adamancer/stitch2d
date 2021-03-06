"""Sorts element maps created by NSS into element-specific folders"""
import glob
import os
import re
import shutil
import tkinter.filedialog
import tkinter




def _get_name(fn):
    """Identifies elements by parsing filenames from NSS

    A typical file name is of the form
    {specimen_name}\[Grid@{x} {y}\]_Counts_{element}_K_map.tif.

    Args:
        fn (str): filename for NSS element map

    Returns:
        Directory name of the form {specimen_name}_{element}
    """
    re_specimen = re.compile('[^\[]+')
    re_map = re.compile('_([A-z]{1,2})_[A-z+]{1,3}_map')
    specimen = re_specimen.match(os.path.splitext(fn)[0]).group()
    specimen = specimen.capitalize().rstrip('_') + '_'
    # Find name of element in filename
    if re_map.search(os.path.splitext(fn)[0]):
        element = re_map.search(os.path.splitext(fn)[0]).group(1)
        return specimen + element.capitalize()
    elif 'RefGrey' in fn:
        return specimen + 'nbsed'
    elif 'Grey' in fn:
        return specimen + 'bsed'
    else:
        return None


def _get_grid(fn):
    """Extracts non-standard grid names from filename"""
    match = re.search(r'\[(.*?)@\d', fn)
    if match is not None:
        grid = match.group(1)
        if grid != 'Grid':
            return '_' + grid
    return ''



def organize(src_dir=None, dst_dir=None):
    """Organizes maps created by NSS into element-specific folders

    This function is accessible from the command line:
    :code:`stitch2d organize <src_dir> <dst_dir>`

    Args:
        src_dir (str): path to folder containing unsorted tiles
        dst_dir (str): path to folder to which to copy the sorted tiles

    Returns:
        None
    """
    root = tkinter.Tk()
    root.withdraw()

    # Prompt user to select source and destination directories
    initial = os.path.expanduser('~')
    if not src_dir:
        title = ('Please select the directory containing the element maps:')
        src_dir = tkinter.filedialog.askdirectory(parent=root, title=title,
                                            initialdir=os.getcwd())
    print('Source directory is {}'.format(src_dir))
    if not dst_dir:
        title = ('Please select the destination'
                 ' for the organized element maps:')
        dst_dir = tkinter.filedialog.askdirectory(parent=root, title=title,
                                            initialdir=os.getcwd())
    print('Destination directory is {}'.format(dst_dir))
    total = 0
    moved = 0
    exist = 0
    for fp in glob.iglob(os.path.join(src_dir, '*.tif')):
        # Set directory name
        fn = os.path.basename(fp)
        dn = _get_name(fn)
        if dn:
            total += 1
            # Create directory if neccesary
            try:
                os.makedirs(os.path.join(dst_dir + _get_grid(fn), dn))
            except OSError:
                pass
            else:
                print('Creating directory {}...'.format(dn))
            # Move file into proper directory
            src = os.path.join(fp)
            dst = os.path.join(dst_dir + _get_grid(fn), dn)
            try:
                open(dst, 'r')
            except IOError:
                print('Copying {}...'.format(fn))
                try:
                    shutil.copy2(src, dst)
                except IOError:
                    print('Could not write to destination. Out of space?')
                    raise
                else:
                    exist += 1
            else:
                #print '{} already exists!'.format(os.path.basename(dst))
                moved += 1
    print(('{:,} files processed ({:,} moved,'
           ' {:,} already existed)').format(total, exist, moved))
