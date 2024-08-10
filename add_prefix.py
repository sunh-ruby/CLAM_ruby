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
import pandas as pd
def add_prefix(prefix, folder):
    files = glob.glob(folder + '/*')
    for file in files:
        if os.path.isfile(file):
            filename = os.path.basename(file)
            newname = prefix + '_' + filename
            os.rename(file, os.path.join(folder, newname))

def modify_csv(prefix, csv_file_path):
    tsv = pd.read_csv(csv_file_path)
    # find slide_id column and add prefix to each element
    slide_id = tsv['slide_id']
    new_slide_id = []
    for slide in slide_id:
        new_slide_id.append(prefix + '_' + slide)
    tsv['slide_id'] = new_slide_id
    tsv.to_csv(csv_file_path, index=False)
def main():
    if len(sys.argv) != 3:
        print('Usage: python add_prefix.py STRINGNAME foldername')
        sys.exit(1)
    prefix = sys.argv[1]
    folder = sys.argv[2]
    sub_folders = glob.glob(folder + '/*')
    if len(sub_folders) == 0:
        print(f"The folder is not existed or empty or {folder} is actually a file")
    for sub_folder in sub_folders:
        if os.path.isdir(sub_folder):
            add_prefix(prefix, sub_folder)
        else:
            # make sure it is a csv file
            assert sub_folder.endswith('.csv'), 'Not a csv file'
            modify_csv(prefix, sub_folder)
    if os.path.exists(folder) and not os.path.isdir(folder):
        modify_csv(prefix, folder)

if __name__ == '__main__':
    main()
