import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, path_transform, eval_transforms, uni_transforms
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, ResNeXt50_trained, Cancer_region_scorer
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import timm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, cancer_region_scorer=False):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
        cancer_region_scorer: if not False, then the scoring model is used to compute the cancer region score
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, custom_transforms= path_transform())
		# uni_transforms
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	from scipy.special import softmax
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy()
			if cancer_region_scorer != False:
				cancer_scores = cancer_region_scorer(batch)
				cancer_scores = cancer_scores.cpu().numpy()
				cancer_scores = softmax(cancer_scores, axis=1)[:,0]
				asset_dict = {'features': features, 'coords': coords, "cancer_region_scorer": cancer_scores}
			else:
				asset_dict = {'features': features, 'coords': coords}
			
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	#model = resnet50_baseline(pretrained= True)
	model = ResNeXt50_trained(pretrained= True)

	#model =  timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
	model = model.to(device)

	CANCER_SCORING =Cancer_region_scorer(pretrained=True)
	CANCER_SCORING= CANCER_SCORING.to(device)

	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	CANCER_SCORING.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			print('skipped {}'.format(slide_id))
			print('file not found: {}'.format(h5_file_path))
			continue
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path.replace('batch1_',"").replace('batch2_',"").replace('batch3_',"").replace('batch4_',""))
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size, cancer_region_scorer=CANCER_SCORING)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))


# CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir proscia_batch1/ --data_slide_dir /mnt/sda/proscia/slides/ --csv_path proscia_batch1/process_list_autogen.csv --feat_dir proscia_batch1/ --batch_size 512 --slide_ext .svs
# or nohup python extract_features_fp.py --data_h5_dir DATA_ROOT_DIR/proscia_batch1/ --data_slide_dir /mnt/sda/proscia/slides/ --csv_path DATA_ROOT_DIR/proscia_batch1/process_list_autogen.csv --feat_dir DATA_ROOT_DIR/proscia_batch1/ --batch_size 512 --slide_ext .svs &