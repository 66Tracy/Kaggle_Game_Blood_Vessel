from pycocotools import mask as mask_utils
import cv2
import glob
import numpy as np
from PIL import Image
import os
import json

# 传入二分的mask转为坐标点返回
def rle_to_polygon(rle):
    # Assume rle is your RLE-encoded mask
    binary_mask = mask_utils.decode(rle)
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 读取train/mask_mixed/*.png全部的图片
# 0）创建一个json文件
json_path = './train/cx_polygons.jsonl'
# os.mkdir(json_path, exist_ok=True)

# 1）循环整个列表，获取id，依次构建字典
all_masks = glob.glob('/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/mask_mixed/*.png')
for mask_path in all_masks:
    tile_dict = {}

    id = mask_path.split('/')[-1].split('_')[0]
    # print(id)
    
    mask = Image.open(mask_path).convert('L')
    # convert the PIL Image into a numpy array
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of binary masks
    masks = [np.where(mask== obj_ids[i, None, None],1,0) for i in range(len(obj_ids))]
    masks = np.array(masks)

    # 2）将一个list的mask们，依次转化为cords
    annotations = []
    for mask in masks:
        # 一个mask一个字典
        annot = {}
        # (512, 512)
        binary_mask = (mask > 0).astype(np.uint8)
        # print(binary_mask.shape)
        # rle is a 字典
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        # int32, tuple
        contours = rle_to_polygon(rle)
        contours = list(contours)

        cords = []
        for cont in contours[0]:
            cords.append(cont[0].tolist())

        annot['type'] = 'blood_vessel'
        annot['coordinates'] = cords

        # 将一个mask对应的字典加入列表
        annotations.append(annot)

    tile_dict['id'] = id
    tile_dict['annotations'] = annotations

    # 将一个图片的字典转化为json文件
    json_str = json.dumps(tile_dict)

    with open (json_path, 'a') as f:
        f.write(json_str+'\n')
    





