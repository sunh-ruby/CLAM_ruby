import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm
from models.experimental_yolo import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_point
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import create_dataloader
from classifier.models import Img_DataLoader, eval_hs
from models.resnext import Myresnext50

class opt:
    def __init__(self):
        self.weights = '/home/harry/Documents/codes/ruby-yolo/runs/train/yolov733/weights/epoch_099.pt'
        self.img_size = 512
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.nosave = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        self.project = 'runs/detect'
        self.name = 'exp'
        self.exist_ok = False
        self.no_trace = False
        self.classifier_path =  "/home/harry/Documents/codes/PatchML/checkpoints_256_batch0-12-CE-addbenign/model_17_0.9962293741235031.pth"


    # when I call opt() it return a class of opt


class YoloResNeXt:
    def __init__(self, opt):
        self.opt = opt()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size
        self.cell_extractor = Cell_embedding_extractors(checkpoint_path = opt.classifier_path)
        if self.half:
            self.model.half()  # to FP16

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        checkpoint_path = "/home/harry/Documents/codes/PatchML/checkpoints_256_batch0-12-CE-addbenign/model_17_0.9962293741235031.pth"
        

        ### construct the model

    def forward(self, x):
        #img = torch.from_numpy(img).to(device)
        #img = img.half() if half else img.float()  # uint8 to fp16/32
        x /= 255.0  # 0 - 255 to 0.0 - 1.0
        if x.ndimension() == 3:
            x = x.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != x.shape[0] or old_img_h != x.shape[2] or old_img_w != x.shape[3]):
            old_img_b = x.shape[0]
            old_img_h = x.shape[2]
            old_img_w = x.shape[3]
            for i in range(3):
                model(x, augment=opt.augment)[0]


        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            temp_img_list = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # first expand the image with 224/2 each side
                    # then crop the 224 by 224 image
                    img0 = cv2.copyMakeBorder(im0, 112, 112, 112, 112, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #print(xyxy)
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    points = [(int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2))]
                    # crop the 224 by 224 image
                    temp_img = img0[points[0][1]:points[0][1]+224, points[0][0]:points[0][0]+224]
                    temp_img_list.append(temp_img)
                    cell_name = save_path.split('/')[-1].split('.')[0] + f'_{points[0][0]}_{points[0][1]}'
                    temp_img_name.append(cell_name)
        feature_embedding = self.extraction(ckpt_dir = checkpoint_path, X_test = temp_img_list, labels = temp_img_name)

        # the output of feature embedding is the feature of the cell images
        # use mean to compute it as the patch feature
        patch_feature = feature_embedding.mean(axis = 0)
        return patch_feature

    def extraction(self, ckpt_dir, X_test, labels):
        cell_types_df =meta_table(class_number=7)
        assert len(X_test) == len(labels), "Length of X_test and labels should be same"
        Orig_img = Img_DataLoader(array_list= X_test, split='viz',df= cell_types_df,transform =transform_pipeline(), 
                                names= labels)
        shuffle = False
        dataloader = DataLoader(Orig_img, batch_size=1024, num_workers=2, shuffle=shuffle, )

        for i, _batch in enumerate(dataloader):

            if i == 0:
                images = _batch["image"].cuda()
                pred_hidden_layer = My_model.pretrained(images)
            else:
                images = _batch["image"].cuda()
                pred_hidden_layer = torch.cat((pred_hidden_layer, My_model.pretrained(images)), 0)
        return pred_hidden_layer.cpu().detach().numpy()
        






class Cell_embedding_extractors:
    def __init__(self, checkpoint_path):
        resnext50_pretrained = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnext50_32x4d", verbose=False, pretrained=True,
        )
        My_model = Myresnext50(
            my_pretrained_model=resnext50_pretrained, num_classes=7
        )

        checkpoint = torch.load(checkpoint_path)


        checkpoint  = remove_data_parallel(checkpoint)

        My_model.load_state_dict(checkpoint, strict=True)
        

        My_model = My_model.cuda().eval()
        cell_types_df =meta_table(class_number=7)
        self.cell_extractor = My_model.pretrained

    def forward(self, x):
        x = self.cell_extractor(x)
        return x



    




def meta_table(class_number ):
    if class_number ==6:
        cell_types = ['Artifact', 'Epithelial cells', 'NSCC', 'RBC', 'Suspicion', 'WBC']

        cell_types = list(cell_types)
        cell_types.sort()

        cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])# converting type of columns to 'category'
        cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')# Assigning numerical values and storing in another column
        cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes

        enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
        enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())# merge with main df bridge_df on key values
        cell_types_df = cell_types_df.join(enc_df)
    elif class_number ==7:
        cell_types = ['Artifact', "Benign_A", 'Epithelial cells', 'NSCC', 'RBC', 'Suspicion', 'WBC']

        cell_types = list(cell_types)
        cell_types.sort()

        cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])# converting type of columns to 'category'
        cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')# Assigning numerical values and storing in another column
        cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes

        enc = OneHotEncoder(handle_unknown='ignore')# passing bridge-types-cat column (label encoded values of bridge_types)
        enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())# merge with main df bridge_df on key values
        cell_types_df = cell_types_df.join(enc_df)     
    else:
        ValueError(f"the class number is {class_number}. currently there is no such setup")  

    return cell_types_df

        
