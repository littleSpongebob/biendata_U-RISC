
import os
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.nn import init
import logging
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from glob import glob
import cv2
from albumentations import (
    Normalize,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomGamma,
    RandomBrightnessContrast,
    Lambda
)

def to_tensor(x, **kwargs):
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg, index_list=None, preprocessing=None):
        super(TrainSetLoader, self).__init__()
        self.preprocessing = preprocessing
        self.dataset_dir = os.path.join(dataset_dir)
        self.all_file_list = [os.path.splitext(file)[0] for file in sorted(glob(os.path.join(self.dataset_dir, 'imgs', '*.png')))]
        if index_list is None:
            self.file_list = self.all_file_list
        else:
            self.file_list = []
            for index in index_list:
                self.file_list.append(self.all_file_list[index])
            assert len(self.all_file_list)//5 == len(self.file_list)//4, 'the length of train file_list is error'
        self.aug = Compose([RandomSizedCrop(p=1,
                                           min_max_height=(int(cfg.size*1), int(cfg.size*1)),
                                           height=cfg.size,
                                           width=cfg.size),
                            RandomBrightnessContrast(p=0.8),
                            RandomGamma(p=0.8),
                           HorizontalFlip(p=0.5),
                           VerticalFlip(p=0.5),
                           RandomRotate90(p=0.5),
                           Transpose(p=0.5),
                           # OneOf([ElasticTransform(p=1, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
                           #        GridDistortion(p=1),
                           #        OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
                           #        ], p=0.8)
                           # ElasticTransform(p=0.9, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                            # GridDistortion(p=0.8),
                            # OpticalDistortion(p=0.8, distort_limit=0.5, shift_limit=0.5),
                           ], p=1)

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index] + '.png')
        # brg->rgb
        img = img[..., ::-1]
        mask = cv2.imread(self.file_list[index].replace('imgs', 'masks') + '.tiff', 0)
        mask = np.where(mask <= 122, 1, 0)

        augmented = self.aug(image=img, mask=mask)

        img, mask = augmented['image'], augmented['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


    def __len__(self):
        return len(self.file_list)

class ValSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg, index_list=None, preprocessing=None):
        super(ValSetLoader, self).__init__()
        self.preprocessing = preprocessing
        self.dataset_dir = os.path.join(dataset_dir)
        self.all_file_list = [os.path.splitext(file)[0] for file in
                              sorted(glob(os.path.join(self.dataset_dir, 'imgs', '*.png')))]
        if index_list is None:
            self.file_list = self.all_file_list
        else:
            self.file_list = []
            for index in index_list:
                self.file_list.append(self.all_file_list[index])
            assert len(self.all_file_list) // 5 == len(self.file_list) , 'the length of val file_list is error'
    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index] + '.png')
        # brg->rgb
        img = img[..., ::-1]
        mask = cv2.imread(self.file_list[index].replace('imgs', 'masks') + '.tiff', 0)
        mask = np.where(mask <= 122, 1, 0)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
    def __len__(self):
        return len(self.file_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, cfg, preprocessing=None):
        super(TestSetLoader, self).__init__()
        self.preprocessing = preprocessing
        self.dataset_dir = os.path.join(dataset_dir, cfg.problem, 'test', 'imgs')
        self.file_list = [os.path.splitext(file)[0] for file in glob(os.path.join(self.dataset_dir, '*'))]
    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index] + '.png')
        # brg->rgb
        img = img[..., ::-1]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img)
            img = sample['image']

        return {'image': torch.from_numpy(img)}
    def __len__(self):
        return len(self.file_list)

def cal_f1(predict, gt,threshold=0.5):
    if isinstance(predict, torch.Tensor) and isinstance(gt, torch.Tensor):
        if predict.is_cuda:
            predict = predict.cpu()
        if gt.is_cuda:
            gt = gt.cpu()
    predict = np.where(predict > threshold, 1, 0)

    predict = predict.reshape(-1)
    gt = gt.reshape(-1)

    return f1_score(gt, predict)


def save_ckpt(network, path, save_filename):
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = os.path.join(path, save_filename)
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_uniform_(m.weight.data)

def setup_logger(logger_name, log_file, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s_%(lineno)d %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.9, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score