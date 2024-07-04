
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import os
import glob 
import random
import math
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.transforms import Compose,  Normalize
from SSIM import *
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Pad, ToPILImage
import torchvision
from scipy import signal
import cv2


class myBatchNormlize(nn.Module):
    def __init__(self):
        super(myBatchNormlize, self).__init__()
        transforms = [Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        self.transforms = Compose(transforms)

    def forward(self,x):
        for m in range(x.size()[0]):
            x[m,:,:,:] = self.transforms(x[m])
        return x

class myUnormlize(nn.Module):
    def __init__(self):
        super(myUnormlize, self).__init__()

    def forward(self, x):
        x = torch.clamp(x, -1., 1.)
        x = (x + 1) / 2
        return x

def weights_init_kaiming(lyr):
    r"""Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)). \
            clamp_(-0.025, 0.025)
        nn.init.constant(lyr.bias.data, 0.0)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # pdb.set_trace()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class Drecloss_stripe(nn.Module):
    def __init__(self, Drecloss_stripe_weight=1):
        super(Drecloss_stripe, self).__init__()
        self.Drecloss_stripe_weight = Drecloss_stripe_weight

    def forward(self, x, y):
        h_x = x.size()[2]
        h_y = y.size()[2]
        h_tv_x = (x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        h_tv_y = (y[:, :, 1:, :] - y[:, :, :h_y - 1, :])
        L1 = torch.nn.L1Loss()
        Drecloss_stripe = L1(h_tv_x, h_tv_y)
        return self.Drecloss_stripe_weight * Drecloss_stripe

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class TVloss(nn.Module):
    def __init__(self, TVloss_weight=1):
        super(TVloss, self).__init__()
        self.TVloss_weight = TVloss_weight

    def forward(self,x,y):
        h_x = x.size()[2]
        w_x = x.size()[3]
        w_tv_x = (x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        w_tv_y = (y[:, :, :, 1:] - y[:, :, :, :w_x - 1])
        h_tv_x = (x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        h_tv_y = (y[:, :, 1:, :] - y[:, :, :h_x - 1, :])
        MSE = torch.nn.MSELoss()
        TVloss = (MSE(h_tv_x, h_tv_y) + MSE(w_tv_x, w_tv_y))*0.5
        return self.TVloss_weight * TVloss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def HBGM(A,B):
    DWT = DWTForward(J=3, wave='haar').cuda()
    IDWT = DWTInverse(wave='haar').cuda()
    DMT3_yl, DMT3_yh = DWT(A)
    DMT3_yl.zero_()
    for i, tensor in enumerate(DMT3_yh):
        DMT3_yh[i][:, :, 1, :, :].zero_()
    out1 = IDWT((DMT3_yl, DMT3_yh))

    DMT3_yl, DMT3_yh = DWT(B)
    DMT3_yl.zero_()
    for i, tensor in enumerate(DMT3_yh):
        DMT3_yh[i][:, :, 1, :, :].zero_()
    out2 = IDWT((DMT3_yl, DMT3_yh))
    return out1,out2

class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.025,  # weight of ssim and l1 loss
                 compensation=1.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=3):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]
        # print(loss_ms_ssim)

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()

class add_noise(nn.Module):
    def __init__(self, case, noiseIntL):
        super(add_noise, self).__init__()
        self.case = case
        self.noiseIntL = noiseIntL

    def forward(self,img_train):
        noise_S = torch.zeros(img_train.size())
        if self.case == 0:
            # 随机定一个 分布的最大值
            beta = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1],
                                     size=noise_S.size()[0])  # generate a row tensor, the insity of noise
            for m in range(noise_S.size()[0]):
                sizeN_S = noise_S[0, 0, :, :].size()
                noise_col = np.random.normal(0, beta[m], sizeN_S[1])  # row tensor
                S_noise = np.tile(noise_col, (sizeN_S[0], 1))  # flatten
                S_noise = np.expand_dims(S_noise, 0)  # add dim
                S_noise = torch.from_numpy(S_noise)  # to tensor
                noise_S[m, :, :, :] = S_noise  # become primary shape

            imgn_trainC = img_train + noise_S
            imgn_train = torch.clip(imgn_trainC, 0., 1.)


        elif self.case == 1:
            beta1 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta2 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            for m in range(noise_S.size()[0]):
                sizeN_S = noise_S[0, 0, :, :].size()
                A1 = np.random.normal(0, beta1[m], sizeN_S[1])  # 一行向量
                A2 = np.random.normal(0, beta2[m], sizeN_S[1])  # 一行向量
                # flatten
                A1 = np.tile(A1, (sizeN_S[0], 1))
                A2 = np.tile(A2, (sizeN_S[0], 1))
                # add dim
                A1 = np.expand_dims(A1, 0)
                A2 = np.expand_dims(A2, 0)
                # to tensor
                A1 = torch.from_numpy(A1)
                A2 = torch.from_numpy(A2)
                imgn_train_m = A1 + A2 * img_train[m] + img_train[m]
                imgn_train_m_c = torch.clip(imgn_train_m, 0., 1.)
                noise_S[m, :, :, :] = imgn_train_m_c
            imgn_train = noise_S

        elif self.case == 2:
            beta1 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta2 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta3 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            for m in range(noise_S.size()[0]):
                sizeN_S = noise_S[0, 0, :, :].size()
                A1 = np.random.normal(0, beta1[m], sizeN_S[1])  # 一行向量
                A2 = np.random.normal(0, beta2[m], sizeN_S[1])  # 一行向量
                A3 = np.random.normal(0, beta3[m], sizeN_S[1])  # 一行向量
                # 拉伸
                A1 = np.tile(A1, (sizeN_S[0], 1))
                A2 = np.tile(A2, (sizeN_S[0], 1))
                A3 = np.tile(A3, (sizeN_S[0], 1))
                # add dim
                A1 = np.expand_dims(A1, 0)
                A2 = np.expand_dims(A2, 0)
                A3 = np.expand_dims(A3, 0)
                # to tensor
                A1 = torch.from_numpy(A1)
                A2 = torch.from_numpy(A2)
                A3 = torch.from_numpy(A3)
                imgn_train_m = A1 + A2 * img_train[m] + A3 * A3 * img_train[m] + img_train[m]
                imgn_train_m_c = torch.clip(imgn_train_m, 0., 1.)
                noise_S[m, :, :, :] = imgn_train_m_c
            imgn_train = noise_S

        elif self.case == 3:
            beta1 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta2 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta3 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta4 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])

            for m in range(noise_S.size()[0]):
                sizeN_S = noise_S[0, 0, :, :].size()
                A1 = np.random.normal(0, beta1[m], sizeN_S[1])  # 一行向量
                A2 = np.random.normal(0, beta2[m], sizeN_S[1])  # 一行向量
                A3 = np.random.normal(0, beta3[m], sizeN_S[1])  # 一行向量
                A4 = np.random.normal(0, beta4[m], sizeN_S[1])  # 一行向量
                # 拉伸
                A1 = np.tile(A1, (sizeN_S[0], 1))
                A2 = np.tile(A2, (sizeN_S[0], 1))
                A3 = np.tile(A3, (sizeN_S[0], 1))
                A4 = np.tile(A4, (sizeN_S[0], 1))
                # add dim
                A1 = np.expand_dims(A1, 0)
                A2 = np.expand_dims(A2, 0)
                A3 = np.expand_dims(A3, 0)
                A4 = np.expand_dims(A4, 0)
                # to tensor
                A1 = torch.from_numpy(A1).cuda()
                A2 = torch.from_numpy(A2).cuda()
                A3 = torch.from_numpy(A3).cuda()
                A4 = torch.from_numpy(A4).cuda()
                imgn_train_m = A1 + A2 * img_train[m] + A3 * A3 * img_train[m] + A4 * A4 * A4 * img_train[m] + \
                               img_train[m]
                imgn_train_m_c = torch.clip(imgn_train_m, 0., 1.)
                noise_S[m, :, :, :] = imgn_train_m_c
            imgn_train = noise_S.cuda()
        elif self.case == 4:
            beta1 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta2 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta3 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])
            beta4 = np.random.uniform(self.noiseIntL[0], self.noiseIntL[1], size=noise_S.size()[0])

            for m in range(noise_S.size()[0]):
                sizeN_S = noise_S[0, 0, :, :].size()
                A1 = np.random.normal(0, beta1[m], sizeN_S[1])  # 一行向量
                A2 = np.random.normal(0, beta2[m], sizeN_S[1])  # 一行向量
                A3 = np.random.normal(0, beta3[m], sizeN_S[1])  # 一行向量
                A4 = np.random.normal(0, beta4[m], sizeN_S[1])  # 一行向量
                # 拉伸
                A1 = np.tile(A1, (sizeN_S[0], 1))
                A2 = np.tile(A2, (sizeN_S[0], 1))
                A3 = np.tile(A3, (sizeN_S[0], 1))
                A4 = np.tile(A4, (sizeN_S[0], 1))
                # add dim
                A1 = np.expand_dims(A1, 0)
                A2 = np.expand_dims(A2, 0)
                A3 = np.expand_dims(A3, 0)
                A4 = np.expand_dims(A4, 0)
                # to tensor
                A1 = torch.from_numpy(A1)
                A2 = torch.from_numpy(A2)
                A3 = torch.from_numpy(A3)
                A4 = torch.from_numpy(A4)
                imgn_train_m = A1 + A2 * img_train[m] + A3 * A3 * img_train[m] + A4 * A4 * A4 * img_train[m] + \
                               img_train[m]
                # imgn_train_m_c = torch.clip(imgn_train_m, 0., 1.)
                noise_S[m, :, :, :] = imgn_train_m
            imgn_train = noise_S
        return imgn_train
