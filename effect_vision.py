
import glob
import tifffile as tiff
import numpy as np
from PIL import Image
import cv2
import os

# 1) 根据路径读取所有照片
# 2）将模型对应的mask输出到对应的文件夹

# 所有图片路径
all_path = glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/Dataset/sparse_labeled/*')
# 所有图片对应的id
all_ids = [impath.split('/')[-1].split('.')[0] for impath in all_path]

prefix = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/Dataset/train/'
save_effect = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/effect/effect_compare/effect_torchvision_sparse/'

# 接口 1）从torchvision获取模型
import torch
import torchvision
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from skimage.morphology import binary_dilation

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
        
        name = self.name_indices[idx]
        img_path = prefix + str(name) +'.tif'
        array = tiff.imread(img_path)
        img = Image.fromarray(array)
        
        img, _ = self.transforms(img, img)

        return img, name

    def __len__(self):
        return len(self.imgs)
    
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

# 从detection_wheel包获取变换工具
from detection_wheel import transforms as T
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

import matplotlib.pyplot as plt
def log_image(id, mask, save_path):

    # 读取原图
    img = cv2.imread('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/Dataset/train/' + id[0] + '.tif')

    last_mask = mask.astype(np.uint8)

    # Convert binary masks to RGB for visualization
    mask_rgb = cv2.cvtColor(last_mask, cv2.COLOR_GRAY2RGB)

    # 给mask染色
    mask_rgb[..., 0] = 0  # zero out the green channel
    mask_rgb[..., 1] = 0  # zero out the blue channel
    mask_rgb = mask_rgb * 255  # scale mask to [0, 255]

    # Overlay the masks on the original image
    overlayed_img = cv2.addWeighted(img, 0.5, mask_rgb, 0.5, 0)

    cv2.imwrite(f"{save_path}{id[0]}.tif",overlayed_img)


# 3) 加载模型参数 - 此处路径须改成训练模型所在
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model_instance_segmentation(num_classes=2)
model.to(device)
model.load_state_dict(torch.load('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/output/well_tuned_weight.pth'))
model.eval()


# 4）加载测试图片 - 根据路径只获取id，然后到dataset train里取图
dataset_test = PennFudanDataset(all_path, get_transform(train=False))
test_dl = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

sample = None
num = 1
with torch.no_grad():
    for img, idx in test_dl:
        img = img.to(device)
        pred = model(img)
        if sample is None: sample=pred
        pred_string = ''
        # print(len(pred[0]['masks']))
        
        img_mask = np.zeros((512,512))
        right = 0
        for m in range(len(pred[0]['masks'])):
            mask = pred[0]['masks'][m].detach().permute(1,2,0).cpu().numpy()
            mask = np.where(mask>0.5, 1, 0).astype(bool)
            mask = binary_dilation(mask)
            
            mask = np.squeeze(mask.astype(int), axis=2)
            score = pred[0]['scores'][m].detach().cpu().numpy()

            # 得到一系列mask
            if score>=0.9:
                img_mask += mask
                right += 1
            else:
                break
        
        # 以防出现重叠
        img_mask = np.where(img_mask>0, 1, 0)
        
        log_image(idx, img_mask, save_effect)