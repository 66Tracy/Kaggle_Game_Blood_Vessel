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

# 这个函数是将输出的mask转化为提交格式
def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str


# 构建dataloader的数据集
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.img_ids = [impath.split('/')[-1].split('.')[0] for impath in imgs]
        
    def __getitem__(self, idx):
        # load id
        img_id = self.img_ids[idx]

        # 原图路径
        img_path = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/Dataset/train/'+img_id+'.tif'

        # 读取图片
        array = tiff.imread(img_path)
        img = Image.fromarray(array)
        
        # 变换图片
        img, _ = self.transforms(img, img)

        # 读取gt
        gt_path = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/medsam/medsam_dataset/val/'+img_id+'.npz'
        with np.load(gt_path) as content:
            gts = content['gts']
        # 合并gt
        img_gt = np.zeros((512,512))
        for i in range(gts.shape[0]):
            img_gt += gts[i]
        
        img_gt = np.where(img_gt>0, 1, 0)

        return img, img_gt, img_id

    def __len__(self):
        return len(self.img_ids)


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


from pycocotools import mask as mask_utils
# 计算单张IoU
def single_image_iou(mask, gt):
    rle1 = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle2 = mask_utils.encode(np.asfortranarray(gt.detach().cpu().numpy().astype(np.uint8)))
    iou = mask_utils.iou([rle1], [rle2], [False])
    return iou

def log_image(id, mask, gt, save_path='./output/'):

    # 读取原图
    img = cv2.imread('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/Dataset/train/' + id[0] + '.tif')

    last_mask = mask

    last_gts = gt
    
    last_mask = last_mask.astype(np.uint8)
    # print(np.where(last_mask[0]>0))
    last_gts = last_gts.detach().cpu().numpy().astype(np.uint8)
    # print(np.where(last_gts[0]>0))

    # Convert binary masks to RGB for visualization
    mask_rgb = cv2.cvtColor(last_mask, cv2.COLOR_GRAY2RGB)
    mask_rgb_gt = cv2.cvtColor(last_gts, cv2.COLOR_GRAY2RGB)

    # 给mask染色
    mask_rgb[..., 1] = 0  # zero out the green channel
    mask_rgb[..., 2] = 0  # zero out the blue channel
    mask_rgb = mask_rgb * 255  # scale mask to [0, 255]

    mask_rgb_gt[..., 0] = 0  # zero out the red channel
    mask_rgb_gt[..., 1] = 0  # zero out the green channel
    mask_rgb_gt = mask_rgb_gt * 255  # scale mask to [0, 255]

    # Overlay the masks on the original image
    overlayed_img = cv2.addWeighted(img, 0.5, mask_rgb, 0.5, 0)
    overlayed_img_gt = cv2.addWeighted(img, 0.5, mask_rgb_gt, 0.5, 0)

    # 定义坐标图面
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))

    ax[0].imshow(img)
    ax[0].title.set_text('Input Images')

    ax[1].imshow(overlayed_img_gt)
    ax[1].title.set_text('Masks by gt')

    ax[2].imshow(overlayed_img)
    ax[2].title.set_text('Masks by mm')

    plt.savefig(f"{save_path}test_miou.eps")


# 3) 加载模型参数 - 此处路径须改成训练模型所在
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)
model.to(device)
model.load_state_dict(torch.load('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/output/well_tuned_weight.pth'))
model.eval()


# 4）加载测试图片 - 此处路径需要改成test所在路径
all_imgs = glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/medsam/medsam_dataset/val/*.npz')
dataset_test = PennFudanDataset(all_imgs, get_transform(train=False))

test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)



# 5）转化成提交格式
sample = None
iou = 0
with torch.no_grad():
    for img, img_gt, img_id in test_dl:
        # print(img_id)
        # print(img.shape, img_gt.shape)
        img = img.to(device)
        pred = model(img)
        if sample is None: sample=pred
        pred_string = ''
        img_mask = np.zeros((512,512))

        for m in range(len(pred[0]['masks'])):
            mask = pred[0]['masks'][m].detach().permute(1,2,0).cpu().numpy()
            mask = np.where(mask>0.5, 1, 0).astype(bool)
            mask = binary_dilation(mask)
            
            # 画框，追加
            # (512, 512, 1) -> (512, 512)
            mask = np.squeeze(mask, axis=2)

            # 得到一系列mask
            if pred[0]['scores'][m].detach().cpu().numpy()>=0.8:
                img_mask += mask.astype(int)
            
        
        # 以防出现重叠
        img_mask = np.where(img_mask>0, 1, 0)
        iou += single_image_iou(img_mask, img_gt[0])

        if img_id[0] == '2b7c5682bb5c':
            log_image(img_id, img_mask, img_gt[0])

print(iou/len(test_dl))
