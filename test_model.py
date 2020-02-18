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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import ttach as tta
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='efficientnet-b7_cls_2')
    parser.add_argument("--ENCODER", type=str, default='efficientnet-b7')
    parser.add_argument("--ENCODER_WEIGHTS", type=str, default='None')
    parser.add_argument('--dataset_dir', type=str, default='/home/sjw/Desktop/U-RISC/pytorch-code/data')
    parser.add_argument("--problem", type=str, default='simple')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--result_path', type=str, default='./test_result')

    parser.add_argument('--test_model_path', type=str, default='/home/sjw/Desktop/U-RISC/pytorch-code/log/efficientnet-b7_cls_2/ckpt/best_ckpt.pth')
    return parser.parse_args()

def test(net, test_loader, cfg):
    cudnn.benchmark = True
    net.eval()
    with torch.no_grad():
        for idx_iter, data in tqdm(enumerate(test_loader)):
            imgs = data['image']
            imgs = imgs.to(cfg.device, dtype=torch.float32)
            imgs_name = test_loader.dataset.file_list[idx_iter].split('/')[-1]

            predict_masks = net(imgs)
            predict_masks = torch.sigmoid(predict_masks)

            ## save result
            if not os.path.exists(os.path.join(cfg.result_path, cfg.name)):
                os.mkdir(os.path.join(cfg.result_path, cfg.name))

            predict_masks = np.array(torch.squeeze(predict_masks.cpu(), 0))
            predict_masks = predict_masks.transpose((1, 2, 0))
            np.save(os.path.join(cfg.result_path, cfg.name, imgs_name), predict_masks)
            predict_masks = np.where(predict_masks > 0.5, 0, 255)

            cv2.imwrite(os.path.join(cfg.result_path, cfg.name, imgs_name + '.tiff'), predict_masks)

if __name__ == '__main__':
    cfg = parse_args()
    if not os.path.exists(os.path.join(cfg.result_path, cfg.name)):
        os.mkdir(os.path.join(cfg.result_path, cfg.name))
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
            tta.Rotate90(angles=[0, 90, 180, 270]),

        ]
    )
    net = tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, 'imagenet')
    test_set = TestSetLoader(dataset_dir=cfg.dataset_dir, cfg=cfg, preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test(net, test_loader, cfg)

