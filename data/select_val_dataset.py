# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  pytorch-code
   File Name    :  select_val_dataset
   Author       :  sjw
   Time         :  20-1-5 16:18
   Description  :  随机选择20%的训练集当做测试集
-------------------------------------------------
   Change Activity: 
                   20-1-5 16:18
-------------------------------------------------
"""
from glob import glob
import shutil
import os
import random
from tqdm import tqdm

random.seed(1)
file_name_list = [(name.split('/')[-1]).split('.')[0] for name in glob(os.path.join('./data/complex', 'train', 'imgs', '*.png'))]
random.shuffle(file_name_list)
src = './data/complex/train'
dst = './data/complex/val'
for file in tqdm(file_name_list[:len(file_name_list)//5]):
    # imgs
    imgs_src = os.path.join(src, 'imgs', file+'.png')
    imgs_dst = os.path.join(dst, 'imgs', file + '.png')
    shutil.move(src=imgs_src, dst=imgs_dst)
    # masks
    masks_src = os.path.join(src, 'masks', file + '.tiff')
    masks_dst = os.path.join(dst, 'masks', file + '.tiff')
    shutil.move(src=masks_src, dst=masks_dst)
