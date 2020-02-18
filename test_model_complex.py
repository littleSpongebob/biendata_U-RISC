# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name :  warp_sr
   File Name    :  test_model
   Author       :  sjw
   Time         :  19-12-20 12:47
   Description  :  
-------------------------------------------------
   Change Activity: 
                   19-12-20 12:47
-------------------------------------------------
"""
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
from torchvision import transforms
import os
import logging
from arch.unet import UNet
import segmentation_models_pytorch as smp
import ttach as tta
from tqdm import tqdm
import numpy as np
from skimage import io
import cv2
from PIL import Image
from glob import glob
# from keras.utils import to_categorical
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='dice_alpha_2_gamma_2.5_inceptionresnetv2_RangerLars_accumulation_steps5_smooth0.01')
    parser.add_argument("--ENCODER", type=str, default='inceptionresnetv2')
    parser.add_argument("--ENCODER_WEIGHTS", type=str, default='None')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument("--problem", type=str, default='complex')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--result_path', type=str, default='./test_result_complex')
    parser.add_argument('--test_model_path', type=str, default='./log_complex/dice_alpha_2_gamma_2.5_inceptionresnetv2_RangerLars_accumulation_steps5_smooth0.01/ckpt/best_ckpt.pth')
    parser.add_argument("--unit", type=int, default=2048, help="the size of the unit img")
    parser.add_argument("--step", type=int, default=1800, help="the size of step")
    parser.add_argument("--pad", type=int, default=100, help="the size of pad")
    parser.add_argument("--GPU_id", type=int, default=1, help="")
    return parser.parse_args()

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

if __name__ == '__main__':
    cfg = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(cfg.GPU_id)
    print('use gpu_{}'.format(cfg.GPU_id))
    if not os.path.exists(os.path.join(cfg.result_path, cfg.name)):
        os.makedirs(os.path.join(cfg.result_path, cfg.name))
    setup_logger('base', os.path.join(cfg.result_path, cfg.name,'{}.log'.format(cfg.name)), level=logging.INFO,
                 screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(cfg)
    if cfg.ENCODER_WEIGHTS == 'None':
        logger.info('set cfg.ENCODER_WEIGHTS = None')
        cfg.ENCODER_WEIGHTS = None
    # net = UNet(n_channels=3, n_classes=1)
    net = smp.Unet(
        encoder_name=cfg.ENCODER,
        encoder_weights=cfg.ENCODER_WEIGHTS,
        classes=1,
        activation=None,
        decoder_attention_type='scse',
        encoder_depth=5,
        decoder_channels=[1024, 512, 256, 128, 64],
        decoder_use_batchnorm=True
    )
    net.to(cfg.device)
    logger.info('load from {}'.format(cfg.test_model_path))
    pretrained_dict = torch.load(cfg.test_model_path)
    net.load_state_dict(pretrained_dict)
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 90, 180, 270]),

        ]
    )
    net = tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, 'imagenet')
    preprocessing = get_preprocessing(preprocessing_fn)

    result_dir = os.path.join(cfg.result_path, cfg.name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    cudnn.benchmark = True
    net.eval()
    with torch.no_grad():
        file_name_list = sorted([(name.split('/')[-1]).split('.')[0] for name in glob(os.path.join(cfg.dataset_dir,
                                                                                            cfg.problem, 'test', 'imgs','*.png'))])

        file_name_list = file_name_list[len(file_name_list)//2*cfg.GPU_id: len(file_name_list)//2*cfg.GPU_id + len(file_name_list)//2]

        for file_name in tqdm(file_name_list):
            print(file_name)
            img = open_big_pic(os.path.join(cfg.dataset_dir,cfg.problem, 'test', 'imgs', '{}.png'.format(file_name)))
            img = np.pad(img, ((cfg.pad, cfg.pad), (cfg.pad, cfg.pad), (0, 0)), mode='constant', constant_values=0)
            h, w, c = img.shape
            label = np.zeros((h, w, 1))
            count_mask = np.zeros((h, w, 1), dtype='uint8')
            h_index = np.arange(0, h - cfg.unit, cfg.step)
            h_index = np.append(h_index, h - cfg.unit)
            w_index = np.arange(0, w - cfg.unit, cfg.step)
            w_index = np.append(w_index, w - cfg.unit)
            k = 0
            for i in h_index:
                for j in w_index:
                    k = k + 1
                    print('\rpredict {}/{} unit image'.format(k, len(h_index)*len(w_index)), end='', flush=True)
                    img_unit = img[i:i + cfg.unit, j:j + cfg.unit, :]
                    img_unit = preprocessing(image=img_unit)['image']
                    img_unit = np.expand_dims(img_unit, axis=0)
                    img_unit = torch.from_numpy(img_unit)
                    img_unit = img_unit.to(cfg.device, dtype=torch.float32)
                    predict_masks_unit = net(img_unit)
                    predict_masks_unit = torch.sigmoid(net(img_unit))
                    predict_masks_unit = np.array(torch.squeeze(predict_masks_unit.cpu(), 0))
                    predict_masks_unit = predict_masks_unit.transpose((1, 2, 0))
                    # np.save(os.path.join(cfg.result_path, cfg.name, imgs_name), predict_masks)
                    # predict_masks_unit = np.where(predict_masks_unit > 0.5, 1, 0)
                    predict_masks_unit = predict_masks_unit[cfg.pad:-cfg.pad, cfg.pad:-cfg.pad, :]
                    label[i + cfg.pad:i - cfg.pad + cfg.unit, j + cfg.pad:j - cfg.pad + cfg.unit, :] = label[
                                                                                                       i + cfg.pad:i - cfg.pad + cfg.unit,
                                                                                                       j + cfg.pad:j - cfg.pad + cfg.unit,:] + \
                                                                                                       predict_masks_unit
                    count_mask[i + cfg.pad:i - cfg.pad + cfg.unit, j + cfg.pad:j - cfg.pad + cfg.unit, :] = count_mask[
                                                                                                       i + cfg.pad:i - cfg.pad + cfg.unit,
                                                                                                       j + cfg.pad:j - cfg.pad + cfg.unit,:] + 1
            label = label / count_mask
            label = label[cfg.pad:-cfg.pad, cfg.pad:-cfg.pad, :]
            if not os.path.exists(os.path.join(result_dir, 'npy')):
                os.makedirs(os.path.join(result_dir, 'npy'))
            np.save(os.path.join(result_dir,'npy', file_name), label)

            # label = label.astype(np.uint8)
            # label = np.argmax(label, axis=2).astype(np.uint8)
            label = np.where(label > 0.5, 0, 255)
            if not os.path.exists(os.path.join(result_dir, 'result')):
                os.makedirs(os.path.join(result_dir, 'result'))
            cv2.imwrite(os.path.join(result_dir, 'result', file_name+'.tiff'), label)



