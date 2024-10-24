import os
import numpy as np
import pandas as pd
import json
import glob

def load_image_dirs():
    folder_name = 256

    img_root_dir = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch0/cells/{folder_name}/"
    )
    img_root_dir_1 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch1/cells/{folder_name}/"
    )
    img_root_dir_2 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch2/cells/{folder_name}/"
    )
    img_root_dir_3 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch3/cells/{folder_name}/"
    )
    img_root_dir_4 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch4/cells/{folder_name}/"
    )
    img_root_dir_5 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch5/cells/{folder_name}/"
    )
    img_root_dir_6 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch6/cells/{folder_name}/"
    )
    img_root_dir_7 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch7/cells/{folder_name}/"
    )
    img_root_dir_8 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch8/cells/{folder_name}/"
    )
    img_root_dir_9 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch9/cells/{folder_name}/"
    )   

    img_root_dir_10 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch10/cells/{folder_name}/"
    )
    img_root_dir_11 = (
        f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch11/cells/{folder_name}/"
    )



    # based on the img_root_dir, the data_info csv file will be saved to the same directory as the img_root_dir

    # first create the data_info csv file with fpath and label columns
    image_dirs_0 = glob.glob(os.path.join(img_root_dir, "*", "*.png"))
    image_dirs_1 = glob.glob(os.path.join(img_root_dir_1, "*", "*.png"))
    image_dirs_2 = glob.glob(os.path.join(img_root_dir_2, "*", "*.png"))
    image_dirs_3 = glob.glob(os.path.join(img_root_dir_3, "*", "*.png"))
    image_dirs_4 = glob.glob(os.path.join(img_root_dir_4, "*", "*.png"))
    image_dirs_5 = glob.glob(os.path.join(img_root_dir_5, "*", "*.png"))
    image_dirs_6 = glob.glob(os.path.join(img_root_dir_6, "*", "*.png"))
    image_dirs_7 = glob.glob(os.path.join(img_root_dir_7, "*", "*.png"))
    image_dirs_8 = glob.glob(os.path.join(img_root_dir_8, "*", "*.png"))
    image_dirs_9 = glob.glob(os.path.join(img_root_dir_9, "*", "*.png"))
    image_dirs_10 = glob.glob(os.path.join(img_root_dir_10, "*", "*.png"))
    image_dirs_11 = glob.glob(os.path.join(img_root_dir_11, "*", "*.png"))
    print("Batch 0: ", len(image_dirs_0))
    print("Batch 1: ", len(image_dirs_1))
    print("Batch 2: ", len(image_dirs_2))
    print("Batch 3: ", len(image_dirs_3))
    print("Batch 4: ", len(image_dirs_4))
    print("Batch 5: ", len(image_dirs_5))
    print("Batch 6: ", len(image_dirs_6))
    print("Batch 7: ", len(image_dirs_7))
    print("Batch 8: ", len(image_dirs_8))
    print("Batch 9: ", len(image_dirs_9))
    print("Batch 10: ", len(image_dirs_10))
    print("Batch 11: ", len(image_dirs_11))


    image_dirs = (
        image_dirs_0
        + image_dirs_1
        + image_dirs_2
        + image_dirs_3
        + image_dirs_4
        + image_dirs_5
        + image_dirs_6
        + image_dirs_7
        + image_dirs_8
        + image_dirs_9
        + image_dirs_10
        + image_dirs_11
    )



    patch_root_dir_0 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch0/raw_images/*')
    patch_root_dir_1 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch1/raw_images/*')
    patch_root_dir_2 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch2/raw_images/*')
    patch_root_dir_3 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch3/raw_images/*')
    patch_root_dir_4 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch4/raw_images/*')
    patch_root_dir_5 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch5/raw_images/*')
    patch_root_dir_6 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch6/raw_images/*')
    patch_root_dir_7 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch7/raw_images/*')
    patch_root_dir_8 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch8/raw_images/*')
    patch_root_dir_9 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch9/raw_images/*')
    patch_root_dir_10 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch10/raw_images/*')
    patch_root_dir_11 = glob.glob('/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch11/raw_images/*')
    # fadcb0e30a9447e59bb7b60836f9_162_132.png
    patch_root_dir = patch_root_dir_0 + patch_root_dir_1 + patch_root_dir_2 + patch_root_dir_3 + patch_root_dir_4 + patch_root_dir_5 + patch_root_dir_6 + patch_root_dir_7 + patch_root_dir_8 + patch_root_dir_9 + patch_root_dir_10 + patch_root_dir_11

    return image_dirs, patch_root_dir


