# for a folder like this:
# proscia_batch3/
#h5_files  masks  patches  process_list_autogen.csv  pt_files  stitches
# in each folder there is a file named as ID. extension
# give a string name STRINGNAME
# put the string name in front of each file in the folder with _ as separator
# for example: STRINGNAME_ID.extension
# usage: python add_prefix.py STRINGNAME foldername

import os
import sys
import glob
def add_prefix(prefix, folder):
    files = glob.glob(folder + '/*')
    for file in files:
        if os.path.isfile(file):
            filename = os.path.basename(file)
            newname = prefix + '_' + filename
            os.rename(file, os.path.join(folder, newname))


def main():
    if len(sys.argv) != 3:
        print('Usage: python add_prefix.py STRINGNAME foldername')
        sys.exit(1)
    prefix = sys.argv[1]
    folder = sys.argv[2]
    sub_folders = glob.glob(folder + '/*')
    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            add_prefix(prefix, sub_folder)
        else:
            pass

if __name__ == '__main__':
    main()
    