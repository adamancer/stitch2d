import glob
import os
import re
import shutil
import tkFileDialog
import Tkinter




def get_name(fn):
    """Identify elements by examining filenames from NSS"""
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




def organizer():
    """Organizes maps created by NSS by element"""
    root = Tkinter.Tk()
    root.withdraw()

    # Prompt user to select source and destination directories
    initial = os.path.expanduser('~')
    title = ("Please select the directory containing the element maps:")
    src_dir = tkFileDialog.askdirectory(parent=root, title=title,
                                        initialdir=initial)
    print 'Source directory is {}'.format(src_dir)
    title = ("Please select the destination for the organized element maps:")
    dst_dir = tkFileDialog.askdirectory(parent=root, title=title,
                                        initialdir=initial)
    print 'Destination directory is {}'.format(dst_dir)
    for fp in glob.iglob(os.path.join(src_dir, '*.tif')):
        # Set directory name
        fn = os.path.basename(fp)
        dn = get_name(fn)
        if dn:
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
                #print '{} already exists!'.format(os.path.basename(dst))
                pass
    print 'Done!'
