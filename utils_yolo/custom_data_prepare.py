import os
import numpy as np
import pandas as pd
import json
import glob
import sys
sys.path.append('/home/harry/Documents/codes/ruby-yolo/')
from utils.load_image_dirs import load_image_dirs

image_dirs, patch_root_dir = load_image_dirs()

labels = [x.split("/")[-2] for x in image_dirs]
labels = ['Suspicious' if x=='Suspicion'  else x for x in labels ]



# how many slides have we annotated, the slide_id is x.split('/')[-1].split('_')[0]
slide_ids = [x.split("/")[-1].split("_")[0] for x in image_dirs]
slide_ids = list(set(slide_ids))
print("Number of slides: ", len(slide_ids))

# all the cell types
cell_types = list(set(labels))
# create a table, per slide, how many images are there, format slideid_patchx_patchyx{x}y{y}.png
number_of_patch = []
cell_type_per_patch = []
patch_names = []
slide_names = []
import tqdm 
for _slide_id in tqdm.tqdm(slide_ids):
    
    img_dirs_per_slide = [
        x for x in image_dirs if x.split("/")[-1].split("_")[0] == _slide_id
    ]
    
    
    patch_list = [
        x.split("/")[-1].split("_")[1]
        + "_"
        + x.split("/")[-1].split("_")[2].split("x")[-2]
        for x in img_dirs_per_slide
    ]
    patch_list = list(set(patch_list))
    
    for patch in patch_list:
        # move the patch to the "/home/harry/Documents/codes/ruby-yolo/cyto/images"
        # raise notimplementederror
        #NotImplementedError('This code is not working yet, do not run it')
        # the new image name is slideid_patchid.png
        batch_dirs = glob.glob(f"/home/harry/Documents/Data/ActivateLearning/proscia_cell/batch*/raw_images/{_slide_id}_{patch}.png")
        # find the patch_dir
        for _dir in batch_dirs:
            if _slide_id +'_'+ patch + '.png' in _dir:
                patch_dir = _dir
                break
        os.system(f"cp {patch_dir} /home/harry/Documents/codes/ruby-yolo/cyto/images/")
        img_dirs_per_patch = [
            x for x in img_dirs_per_slide if patch == x.split("/")[-1].split("_")[1]+ "_"  + x.split("/")[-1].split("_")[2].split("x")[-2] 
        ]
        
        
        with open(f"/home/harry/Documents/codes/ruby-yolo/cyto/labels/{_slide_id}_{patch}.txt", "w") as f:
            if len(img_dirs_per_patch) == 0:
                f.write("")
            else:
                for img_dir in img_dirs_per_patch:
                    center_x_y =     [
                            int(img_dir.split("/")[-1].split("_")[2].split("x")[-1].split("y")[0]),
                            int(img_dir.split("/")[-1].split("_")[2].split("x")[-1].split("y")[-1].split(".")[0]),
                        ]
                    assert center_x_y[0] <512, f'{center_x_y[0]} is bigger than 512'
                    assert center_x_y[1] < 512, f'{center_x_y[1]} is bigger than 512'
                    #print(center_x_y)
                    #print('----------')
                    # convert this to four corners of the image patch, the image size is 96 X 96
                    x_min = (center_x_y[0] - 48)/512
                    x_max = (center_x_y[0] + 48)/512
                    y_min = (center_x_y[1] - 48)/512
                    y_max = (center_x_y[1] + 48)/512
                    #print(y_min, y_max)
                    x_min = max(0,x_min)
                    y_min = max(0, y_min)
                    x_max = min(1, x_max)
                    y_max = min(1, y_max)
                    f.write(f"0 {x_min} {y_min} {x_min} {y_max} {x_max} {y_max} {x_max} {y_min}\n")
                    assert x_min < x_max,  f"the cords are off x_min {x_min} x_max {x_max}"
                    assert y_min < y_max, f"the cords are off y_min {y_min} y_max {y_max}"
            
dirs = glob.glob("/home/harry/Documents/codes/ruby-yolo/cyto/images/train/*")
dirs = [x.split("/")[-1] for x in dirs]

dirs = ["./images/train/" +x for x in dirs]
with open("/home/harry/Documents/codes/ruby-yolo/cyto/train.txt", "w") as f:
    for _dir in dirs:
        f.write(_dir + "\n")
dirs = glob.glob("/home/harry/Documents/codes/ruby-yolo/cyto/images/val/*")
dirs = [x.split("/")[-1] for x in dirs]
dirs = ["./images/val/" +x for x in dirs]
with open("/home/harry/Documents/codes/ruby-yolo/cyto/val.txt", "w") as f:
    for _dir in dirs:
        f.write(_dir + "\n")