import os
import json
from PIL import Image
from collections import Counter

import numpy as np
import pandas as pd

import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2
import shutil

import glob

with open('../Dataset/polygons.jsonl', 'r') as json_file:
    json_list = list(json_file)


# 用一个列表存储所有的gt数据
tiles_dicts = []
for json_str in json_list:
    tiles_dicts.append(json.loads(json_str))


# 把tile的数据也读取
# 7033行，0 - 7032
# 5列，id, source_wsi, dataset, i, j
tile_meta_df = pd.read_csv("../Dataset/tile_meta.csv")
print(tile_meta_df.info())

# 读1行试试
rows = tile_meta_df.head(7033)
print(rows['id'][0])

# 将读取的信息，"id":dataset构建出字典
id_dataset = {}
for i in range(len(rows)):
    id_dataset[str(rows['id'][i])] = rows['dataset'][i]

# 从/Dataset/train/labeled 读取所有id
# 判断该id在字典里是1还是2，如果是1那么是well labeled，如果是2那么是sparse labeled
path = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/image_mixed/*'
all_labeled = glob.glob(path)

# 所有有label的id
all_ids = [impath.split('/')[-1].split('.')[0] for impath in all_labeled]

d1 = 0
d2 = 0
d3 = 0
for id in all_ids:
    if id_dataset[id] == 1:
        d1 += 1
    elif id_dataset[id] == 2:
        d2 += 1
    elif id_dataset[id] == 3:
        d3 += 1

# 419, 1207, 1991
print(d1, d2, d3)

# prefix = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/image/'
# mask_prefix = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/mask/'
# dst = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/well_labeled/'
# dst2 = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/sparse_labeled/'
# mask_dst = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/mask_labeled/'
# mask_dst2 = '/home/chenxi/kaggle/HuBMAP - Hacking the Human Vasculature/mmdetection/train/mask_sparse/'
# well = 0
# sparse = 0
# for id in all_ids:
#     if id_dataset[id] == 1:
#         shutil.copyfile(prefix+str(id)+'.png', dst+str(id)+'.png')
#         shutil.copyfile(mask_prefix+id+'_mask.png', mask_dst+id+'_mask.png')
#         well += 1
#     elif id_dataset[id] == 2:
#         shutil.copyfile(prefix+str(id)+'.png', dst2+str(id)+'.png')
#         shutil.copyfile(mask_prefix+str(id)+'_mask.png', mask_dst2+str(id)+'_mask.png')
#         sparse += 1

# print(well, sparse)





# # 
# mask = np.zeros((512, 512), dtype=np.float32)
# for annot in tiles_dicts[0]['annotations']:
#     cords = annot['coordinates']
#     if annot['type'] == "blood_vessel":
#         for cd in cords:
#             rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
#             mask[rr, cc] = 1
            
# plt.imshow(mask)
# plt.show()


# contours,_ = cv2.findContours((mask*255).astype(np.uint8), 1, 2)
# zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")
# for p in contours:
#     cv2.fillPoly(zero_img, [p], (255, 255, 255))
# plt.imshow(zero_img)
# plt.show()




# from copy import deepcopy
# contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# img_with_area = zero_img
# print(img_with_area.shape)

# print(len(contours))
        
# for i in range(len(contours)):
#     if cv2.contourArea(contours[i]) > (mask.shape[0] * mask.shape[1]) * 0.0001:
#         cv2.fillPoly(img_with_area, [contours[i][:,0,:]], (255-4*(i+1),255-4*(i+1),255-4*(i+1)), lineType=cv2.LINE_8, shift=0)
        
# plt.imshow(img_with_area)
# plt.show()



# def make_seg_mask(tiles_dict):
#     mask = np.zeros((512, 512), dtype=np.float32)
#     for annot in tiles_dict['annotations']:
#         cords = annot['coordinates']
#         if annot['type'] == "blood_vessel":
#             for cd in cords:
#                 rr, cc = np.array([i[1] for i in cd]), np.asarray([i[0] for i in cd])
#                 mask[rr, cc] = 1
                
#     contours,_ = cv2.findContours((mask*255).astype(np.uint8), 1, 2)
#     zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")

#     for p in contours:
#         cv2.fillPoly(zero_img, [p], (255, 255, 255))

#     contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     img_with_area = zero_img

#     for i in range(len(contours)):
#         cv2.fillPoly(img_with_area, [contours[i][:,0,:]], (255-4*(i+1),255-4*(i+1),255-4*(i+1)), lineType=cv2.LINE_8, shift=0)
            
#     return img_with_area    

# os.makedirs('train/image', exist_ok=True)
# os.makedirs('train/mask', exist_ok=True)

# for i, tldc in enumerate(tqdm(tiles_dicts)):
#     array = tiff.imread(f'../Dataset/train/{tldc["id"]}.tif')
#     img_example = Image.fromarray(array)
#     img = np.array(img_example)
#     mask = make_seg_mask(tldc)
    
#     if np.sum(mask)>0:
#         cv2.imwrite(f'train/image/{tldc["id"]}.png', img)
        # cv2.imwrite(f'train/mask/{tldc["id"]}_mask.png', mask)
    












