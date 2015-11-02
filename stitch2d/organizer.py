"""Sorts element maps created by NSS into element-specific folders"""

import glob
import os
import re
import shutil
import tkFileDialog
import Tkinter




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

    specimen = (re_specimen.match(os.path.splitext(fn)[0])
                .group().capitalize().rstrip('_') + '_')
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
    root = Tkinter.Tk()
    root.withdraw()

    # Prompt user to select source and destination directories
    initial = os.path.expanduser('~')
    if not src_dir:
        title = ('Please select the directory containing the element maps:')
        src_dir = tkFileDialog.askdirectory(parent=root, title=title,
                                            initialdir=initial)
    print 'Source directory is {}'.format(src_dir)
    if not dst_dir:
        title = ('Please select the destination'
                 ' for the organized element maps:')
        dst_dir = tkFileDialog.askdirectory(parent=root, title=title,
                                            initialdir=initial)
    print 'Destination directory is {}'.format(dst_dir)
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
                os.mkdir(os.path.join(dst_dir, dn))
            except OSError:
                pass
            else:
                print 'Creating directory {}...'.format(dn)
            # Move file into proper directory
            src = os.path.join(src_dir, fn)
            dst = os.path.join(dst_dir, dn, fn)
            try:
                open(dst, 'rb')
            except IOError:
                print 'Copying {}...'.format(os.path.basename(dst))
                try:
                    shutil.copy2(src, dst)
                except IOError:
                    print 'Could not write to destination. Out of space?'
                    raise
                else:
                    exist += 1
            else:
                #print '{} already exists!'.format(os.path.basename(dst))
                moved += 1
    print ('{:,} files processed ({:,} moved,'
           ' {:,} already existed)').format(total, exist, moved)
