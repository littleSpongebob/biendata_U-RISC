# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  pytorch-code
   File Name    :  cut_data
   Author       :  sjw
   Time         :  20-1-5 15:05
   Description  :  
-------------------------------------------------
   Change Activity: 
                   20-1-5 15:05
-------------------------------------------------
"""
import numpy as np
from skimage import io
import cv2
import os
from PIL import Image
import argparse
from glob import glob

def get_arguments():
    parser = argparse.ArgumentParser(description="crop")
    parser.add_argument("--unit", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default='./data/complex/train')
    parser.add_argument("--image_path", type=str, default='./data/complex/ori_imgs')
    parser.add_argument("--label_path", type=str, default='./data/complex/ori_masks')
    return parser.parse_args()

args = get_arguments()

UNIT = args.unit
OUTPUT_DIR = args.output_dir
IMAGE_PATH = args.image_path
LABEL_PATH = args.label_path

def open_big_pic(path):
    '''
    :param path: 图像路径
    :return: 图像numpy数组
    '''
    Image.MAX_IMAGE_PIXELS = 100000000000
    print('open{}'.format(path))
    img = Image.open(path)   # 注意修改img路径
    img = np.asarray(img)
    print('img_shape:{}'.format(img.shape))
    return img


def crop_img_label(img, label, output_dir, ori_name):
    if not os.path.exists(os.path.join(output_dir, 'imgs')):
        print('output imgs dir not exists make {} dir'.format(output_dir))
        os.makedirs(os.path.join(output_dir, 'imgs'))
    if not os.path.exists(os.path.join(output_dir, 'masks')):
        print('output masks dir not exists make {} dir'.format(output_dir))
        os.makedirs(os.path.join(output_dir, 'masks'))
    # if not os.path.exists(os.path.join(output_dir, 'vis')):
    #     print('output dir not exists make {} dir'.format(output_dir))
    #     os.makedirs(os.path.join(output_dir, 'vis'))
    img = np.pad(img, ((0, 281), (0, 282), (0, 0)), mode='symmetric')
    label = np.pad(label, ((0, 281), (0, 282)), mode='symmetric')
    label = np.where(label>122, 255, 0)
    label_h, label_w = label.shape
    img_h, img_w, _ = img.shape
    assert label_h == img_h
    assert label_w == img_w
    print(label_h, label_w)
    h_index = 0
    k = 0
    while h_index <= label_h - UNIT:
        w_index = 0
        while w_index <= label_w - UNIT:
            img_unit = img[h_index:h_index + UNIT, w_index:w_index + UNIT, :]
            label_unit = label[h_index:h_index + UNIT, w_index:w_index + UNIT]
            if np.sum(np.where(label_unit==0,1,0)) > 100:
                k = k + 1
                print('\rcrop {} unit image'.format(k), end='', flush=True)
                path_unit_img = os.path.join(output_dir, 'imgs', '{}_{}.png'.format(ori_name, k))
                path_unit_label = os.path.join(output_dir, 'masks', '{}_{}.tiff'.format(ori_name, k))
                io.imsave(path_unit_img, img_unit)
                cv2.imwrite(path_unit_label, label_unit)
            w_index = w_index + UNIT
        h_index = h_index + UNIT


if __name__ == '__main__':
    img_path = IMAGE_PATH
    label_path = LABEL_PATH
    img_name = [(name.split('/')[-1]).split('.')[0] for name in glob(os.path.join(img_path, '*.png'))]
    for ori_name in img_name:
        img = open_big_pic(os.path.join(img_path, ori_name+'.png'))
        label = open_big_pic(os.path.join(label_path, ori_name+'.tiff'))
        crop_img_label(img, label, OUTPUT_DIR, ori_name=ori_name)
