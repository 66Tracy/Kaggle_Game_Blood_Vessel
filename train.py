import os, glob
import sys

from PIL import Image

import numpy as np


from sklearn.model_selection import KFold

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models import list_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.resnet import ResNet50_Weights


from detection_wheel import transforms as T


sys.path.append("/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/detection_wheel")


# 提取出图片作数据
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs#sorted(glob.glob('/kaggle/input/hubmap-making-dataset/train/image/*.png'))
        self.masks = masks#sorted(glob.glob('/kaggle/input/hubmap-making-dataset/train/mask/*.png'))

    def __getitem__(self, idx):
        # 打开图片
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        
        # 从路径打开mask
        mask = Image.open(mask_path).convert('L')
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = [np.where(mask== obj_ids[i, None, None],1,0) for i in range(len(obj_ids))]
        masks = np.array(masks)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # print(area,area.shape,area.dtype)
        except:
            area = torch.tensor([[0],[0]])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        #print(masks.shape)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            

        return img, target

    def __len__(self):
        return len(self.imgs)




# 查看有几种候选模型
detection_models = list_models(module=torchvision.models.detection)
print(detection_models)

# 创建模型
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # 这里加载的是一个mask-rcnn + resnet50 + fpn的多层网络提取结构
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT", weights_backbone=ResNet50_Weights.IMAGENET1K_V2)

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

# 模型被默认下载到了这里
# Downloading: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth" 
# to /home/chenxi/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth



# 一个图像变换器
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




from engine import train_one_epoch, evaluate, eval_miou
import utils



device = 'cuda'
# 这个路径只是获取有多少张图片
n_imgs = len(glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/pesudo label/0.449(0.8)0.473(0.7)/image_mixed/*'))

EPOCHS = 4
kf = KFold(n_splits=10, shuffle=True, random_state=43)
for i, (train_index, test_index) in enumerate(kf.split(range(n_imgs))):
    # 训练到第三个fold时就退出了
    if i==3: break
    # 这里才是读取所有的img和mask
    all_imgs = sorted(glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/pesudo label/0.449(0.8)0.473(0.7)/image_mixed/*.png'))
    all_masks = sorted(glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/pesudo label/0.449(0.8)0.473(0.7)/mask_mixed/*.png'))
    all_imgs = np.array(all_imgs)
    all_masks = np.array(all_masks)
    train_img = all_imgs[train_index]
    train_mask = all_masks[train_index]
    val_img = all_imgs[test_index]
    val_mask = all_masks[test_index]
    dataset_train = PennFudanDataset(train_img, train_mask, get_transform(train=True))
    dataset_val = PennFudanDataset(val_img, val_mask, get_transform(train=False))
    train_dl = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True, collate_fn=utils.collate_fn)
    val_dl = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True,collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0003, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(EPOCHS):

        train_one_epoch(model, optimizer, train_dl, device, epoch, print_freq=50)
        evaluate(model, val_dl, device=device)
        scheduler.step()
        
        model_path = f'/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/output_pth/output5-448test/10fold_{i}_epoch{epoch}.pth'
        torch.save(model.state_dict(), model_path)












