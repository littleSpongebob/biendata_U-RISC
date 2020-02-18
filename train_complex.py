from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
import lovasz_losses as L
from torchvision import transforms
import segmentation_models_pytorch as smp
import os
import logging
# from arch.unet import
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
from skimage import io
from tensorboardX import SummaryWriter
import apex
from functools import partial
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='dice_alpha_2_gamma_2.5_RangerLars_accumulation_steps20')
    parser.add_argument("--problem", type=str, default='complex')
    parser.add_argument("--ENCODER", type=str, default='efficientnet-b7')
    parser.add_argument("--ENCODER_WEIGHTS", type=str, default='imagenet')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.3, help='')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--save_val_result', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='RangerLars')
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--dice_alpha', type=float, default=2)

    return parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = 0.99 * targets + 0.01 * (1 - targets)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        # alpha_t = torch.where(targets == 1, torch.tensor(self.alpha).cuda(), torch.tensor(1-self.alpha).cuda())
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss_Fn(nn.Module):
    def __init__(self, sample_wise=False, use_focal=False, apply_nonlin=False, gamma=1.5):
        super(DiceLoss_Fn, self).__init__()
        self.sample_wise = sample_wise
        self.use_focal = use_focal
        self.apply_nonlin = apply_nonlin
        self.act = nn.Sigmoid()
        self.gamma = gamma

    def forward(self, input, label):
        label = 0.99 * label + 0.01 * (1 - label)
        eps = 1e-10
        if self.apply_nonlin:
            input = self.act(input)
        if not self.sample_wise:
            return 1 - (2 * (input * label).sum() + eps) / (input.sum() + label.sum() + eps)
        mul = torch.einsum('nchw->n', input*label)
        sum1 = torch.einsum('nchw->n', input)
        sum2 = torch.einsum('nchw->n', label)
        loss_sample_wise = 1 - (2 * mul / (sum1 + sum2)).mean()
        if not self.use_focal:
            return loss_sample_wise
        return (loss_sample_wise ** self.gamma).mean()

def test(net, test_loader, cfg, iter):
    cudnn.benchmark = True
    net.eval()
    f1_epoch = []
    with torch.no_grad():
        for idx_iter, data in tqdm(enumerate(test_loader)):
            imgs, masks = data['image'], data['mask']
            imgs, masks = imgs.to(cfg.device, dtype=torch.float32), masks.to(cfg.device, dtype=torch.float32)
            imgs_name = test_loader.dataset.file_list[idx_iter].split('/')[-1]

            predict_masks = net(imgs)
            predict_masks = torch.sigmoid(predict_masks)
            f1_epoch.append(cal_f1(predict_masks, masks))
            if cfg.save_val_result:
                ## save results
                if not os.path.exists('log_complex/'+cfg.name + '/iter_{}'.format(iter)):
                    os.mkdir('log_complex/' + cfg.name +'/iter_{}'.format(iter))

                predict_masks = np.array(torch.squeeze(predict_masks.cpu(), 0))
                predict_masks = np.where(predict_masks > 0.5, 0, 255)
                predict_masks = predict_masks.transpose((1, 2, 0))
                cv2.imwrite('log_complex/' + cfg.name +'/iter_{}'.format(iter) + '/' + imgs_name + '.tiff', predict_masks)

        ## print results
        mean_f1 = float(np.array(f1_epoch).mean())
        logger.info('iter_{} mean f1: {}'.format(iter, mean_f1))
        tensorboard_writer.add_scalar('f1', mean_f1, iter)
    net.train()
    return mean_f1


def train(train_loader, cfg):
    # net = UNet(n_channels=3, n_classes=1)
    # net = NestedUNet()
    # net.apply(weights_init_xavier)
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
    # net = convert_model(net)
    net = apex.parallel.convert_syncbn_model(net)
    logger.info(net)
    logger.info('parameters: {}'.format(sum(map(lambda x: x.numel(), net.parameters()))))
    net.to(cfg.device)
    if os.path.exists(cfg.pretrain_path):
        logger.info('load weight from {}'.format(cfg.pretrain_path))
        pretrained_dict = torch.load(cfg.pretrain_path)
        net.load_state_dict(pretrained_dict)
    cudnn.benchmark = True

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5], dtype=torch.float32).cuda())
    criterion = FocalLoss(gamma=cfg.focal_gamma)
    # criterion_1 = DiceLoss_Fn()
    criterion_1 = smp.utils.losses.DiceLoss(activation='sigmoid')

    mom = 0.9
    alpha = 0.99
    eps = 1e-6
    if cfg.optimizer == 'Adam':
        logger.info('use Adam')
        optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'RangerLars':
        logger.info('use RangerLars')
        from over9000.over9000 import Over9000
        optimizer = partial(Over9000, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Novograd':
        logger.info('use Novograd')
        from over9000.novograd import Novograd
        optimizer = partial(Novograd, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Ralamb':
        logger.info('use Ralamb')
        from over9000.ralamb import Ralamb
        optimizer = partial(Ralamb,  betas=(mom,alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'LookaheadAdam':
        logger.info('use LookaheadAdam')
        from over9000.lookahead import LookaheadAdam
        optimizer = partial(LookaheadAdam, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Ranger':
        logger.info('use Ranger')
        from over9000.ranger import Ranger
        optimizer = partial(Ranger,  betas=(mom,alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    else:
        raise NameError
    # if not cfg.use_Radam:
    #     logger.info('use adam')
    #     optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    # else:
    #     logger.info('use Radam')
    #     optimizer = RAdam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    # 半精度
    # net, optimizer = apex.amp.initialize(net, optimizer, opt_level="O1")
    net = nn.DataParallel(net)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.gamma, patience=15, verbose=True)
    loss_epoch = []
    f1_epoch = []
    best_f1 = 0
    iter_per_epoch = len(train_loader)
    accumulation_steps = 20
    for idx_epoch in range(cfg.n_epochs):
        net.train()
        for idx_iter, data in tqdm(enumerate(train_loader)):
            total_idx_iter = idx_epoch * iter_per_epoch + idx_iter + 1
            imgs, masks = data['image'], data['mask']
            imgs, masks = imgs.to(cfg.device, dtype=torch.float32), masks.to(cfg.device,dtype=torch.float32)
            predict_masks = net(imgs)

            loss = criterion(predict_masks, masks) + cfg.dice_alpha * criterion_1(predict_masks, masks)
            # loss = criterion_1(predict_masks, masks)
            # loss = loss / accumulation_steps
            loss.backward()

            if total_idx_iter % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            # with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
                loss_epoch.append(loss.data.cpu())
                f1_epoch.append(cal_f1(torch.sigmoid(predict_masks), masks))

            if total_idx_iter % 100 == 0:

                mean_loss = float(np.array(loss_epoch).mean())
                mean_f1 = float(np.array(f1_epoch).mean())
                logger.info('iter:{:5d} lr:{}, loss:  {:5f}, f1:  {:5f}'.format(total_idx_iter, optimizer.param_groups[0]['lr'],  mean_loss, mean_f1))
                scheduler.step(mean_loss)
                loss_epoch = []
                f1_epoch = []

            if total_idx_iter % 1000 == 0:
                preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, "imagenet")
                test_set = ValSetLoader(dataset_dir=os.path.join(cfg.dataset_dir, cfg.problem, 'val'), cfg=cfg,
                                        preprocessing=get_preprocessing(preprocessing_fn))
                test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
                temp_f1 = test(net, test_loader, cfg, total_idx_iter)
                if temp_f1 > best_f1:
                    save_ckpt(net,
                              path=os.path.join('./log_complex', '{}'.format(cfg.name), 'ckpt'),
                              save_filename='best_ckpt.pth')
                    line = 'best_f1: {} in {} epoch {} iter'.format(temp_f1, idx_epoch + 1, total_idx_iter)
                    filename = os.path.join('./log_complex', '{}'.format(cfg.name), 'ckpt', 'msg.txt')
                    with open(filename, 'w') as f:
                        f.write(line)
                    best_f1 = temp_f1

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, "imagenet")
    test_set = ValSetLoader(dataset_dir=os.path.join(cfg.dataset_dir, cfg.problem, 'val'), cfg=cfg,
                            preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
    temp_f1 = test(net, test_loader, cfg, total_idx_iter)
    if temp_f1 > best_f1:
        save_ckpt(net,
                  path=os.path.join('./log_complex', '{}'.format(cfg.name), 'ckpt'),
                  save_filename='best_ckpt.pth')
        line = 'best_f1: {} in {} epoch {} iter'.format(temp_f1, idx_epoch + 1, total_idx_iter)
        filename = os.path.join('./log_complex', '{}'.format(cfg.name), 'ckpt', 'msg.txt')
        with open(filename, 'w') as f:
            f.write(line)
        best_f1 = temp_f1




if __name__ == '__main__':
    cfg = parse_args()
    if not os.path.exists(os.path.join('./log_complex', '{}'.format(cfg.name))):
        os.makedirs(os.path.join('./log_complex', '{}'.format(cfg.name)))
    setup_logger('base', os.path.join('./log_complex', '{}'.format(cfg.name), '{}.log_complex'.format(cfg.name)), level=logging.INFO,
                       screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(cfg)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, "imagenet")
    train_set = TrainSetLoader(dataset_dir=os.path.join(cfg.dataset_dir, cfg.problem, 'train'), cfg=cfg, preprocessing=get_preprocessing(preprocessing_fn))
    logger.info('total {}, {} iter per epoch'.format(len(train_set), len(train_set) // cfg.batch_size))
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True)

    tensorboard_log_dir = os.path.join('./tensorboard_log_complex', cfg.name)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    if cfg.ENCODER_WEIGHTS == 'None':
        logger.info('set cfg.ENCODER_WEIGHTS = None')
        cfg.ENCODER_WEIGHTS = None
    train(train_loader, cfg)

