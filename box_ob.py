import os, glob
import sys
import json
from PIL import Image
from collections import Counter

import numpy as np
import pandas as pd

import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
from skimage.morphology import binary_dilation

import pandas as pd

from sklearn.model_selection import KFold

# detection_wheel文件夹的绝对路径
sys.path.append("/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/detection_wheel")

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib


# 构建dataloader的数据集
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs
        self.name_indices = [os.path.splitext(os.path.basename(i))[0] for i in imgs]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        name = self.name_indices[idx]
        array = tiff.imread(img_path)
        img = Image.fromarray(array)
        
        img, _ = self.transforms(img, img)

        return img, name

    def __len__(self):
        return len(self.imgs)


# 接口 1）从torchvision获取模型
import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# 2）从detection_wheel包获取变换工具
from detection_wheel import transforms as T
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)


# 3) 加载模型参数 - 此处路径须改成训练模型所在
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)
model.to(device)
model.load_state_dict(torch.load('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/output_pth/output5-448test/10fold_0_epoch3.pth'))
# print(str(model))
model.eval()


# 4）加载测试图片 - 此处路径需要改成test所在路径
all_imgs = glob.glob('../Dataset/test/*.tif')
dataset_test = PennFudanDataset(all_imgs, get_transform(train=False))
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)


# 5）转化成提交格式
ids = []
heights = []
widths = []
prediction_strings = []

sample = None
with torch.no_grad():
    for img, idx in test_dl:
        img = img.to(device)
        
        pred = model(img)
        print(pred[0].keys())

        img = cv2.imread(f'../Dataset/test/{idx[0]}.tif')

        if sample is None: sample=pred
        pred_string = ''
        print(len(pred[0]['masks']))
        
        num = 0
        for m in range(len(pred[0]['masks'])):

            if pred[0]['labels'][m] != 1:
                print("jump")
                continue

            mask = pred[0]['masks'][m].detach().permute(1,2,0).cpu().numpy()
            mask = np.where(mask>=0.5, 1, 0).astype(bool)
            mask = binary_dilation(mask)
            
            score = pred[0]['scores'][m].detach().cpu().numpy()
            box = pred[0]['boxes'][m].detach().cpu().numpy().astype(int)

            # print(box)
            
            if score >= 0.5:
                num += 1
                score = np.around(score,2)
                print(score)
                cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
                cv2.putText(img, str(score), (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imwrite('./Image3.png',img)
        print(num)

            

# 展示test图片
# array = tiff.imread('../Dataset/test/72e40acccadf.tif')
# plt.imshow(array)
# plt.show()

# 如果只有一张图片
# if len(all_imgs)==1:
#     top20 = [sample[0]['masks'][i].cpu().numpy().reshape(512, 512) for i in range(min(20,len(sample[0]['masks'])))]
    
#     pred_img = np.zeros((512,512), dtype=np.float32)
#     for i, j in enumerate(top20):
#         pred_img += j * (1 - 1/len(top20)*i)
#         pred_img = np.clip(pred_img, 0, 1)
#         print(sample[0]['scores'][i].cpu().numpy())
#         # plt.imshow(j)
#         # plt.show()
        
#     plt.imshow(pred_img)
#     plt.show()











