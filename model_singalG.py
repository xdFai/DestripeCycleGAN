import networks
import torch
import torch.nn as nn
import pickle
from utils import *
from models.MWUNet import MWUNet
from SSIM import *


class DerainCycleGAN(nn.Module):
    def __init__(self, opts):
        super(DerainCycleGAN, self).__init__()

        # parameters
        lr = 0.0001

        # discriminators
        self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm,
                                           sn=opts.dis_spectral_norm)

        # generator
        self.genA = MWUNet(3, 3)
        # cubic noise
        case = 3
        noise = [0.02, 0.12]
        self.genB = add_noise(case, noise)
        self.myBatchNormlize = myBatchNormlize().cuda(opts.gpu)
        self.myUnnormlize = myUnormlize().cuda(opts.gpu)

        # vgg
        self.vgg = networks.Vgg16()

        # optimizers
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.genA_opt = torch.optim.Adam(self.genA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        # Setup the loss function for training
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(opts.gan_mode).cuda(opts.gpu)
        self.criterionRGM = GANLoss('lsgan').cuda(opts.gpu)
        self.TVloss = Drecloss_stripe().cuda(opts.gpu)
        self.ms_ssim_mix = MS_SSIM_L1_LOSS().cuda(opts.gpu)
        self.ssimloss = SSIM().cuda(opts.gpu)

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(opts.pool_size)
        self.fake_A1_pool = ImagePool(opts.pool_size)
        self.fake_B_pool = ImagePool(opts.pool_size)

    #   权重初始化
    def initialize(self):
        self.disA.apply(networks.gaussian_weights_init)
        self.genA.apply(networks.gaussian_weights_init)

    # 学习率衰减类型
    def set_scheduler(self, opts, last_ep=0):
        self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
        self.genA_sch = networks.get_scheduler(self.genA_opt, opts, last_ep)

    # 将数据放入GPU
    def setgpu(self, gpu):
        self.gpu = gpu
        self.disA.cuda(self.gpu)
        self.genA.cuda(self.gpu)
        self.genB.cuda(self.gpu)
        self.vgg.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):  #
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def test_forward(self, image1, image2=None, a2b=None):
        if a2b:
            self.fake_A_encoded = self.genA.forward(image1)
        return self.fake_A_encoded

    def forward(self, ep, opts):
        '''self.real_A_encoded -> self.fake_A_encoded -> self.real_A_recon'''
        '''self.real_B_encoded -> self.fake_B_encoded -> self.real_B_recon'''
        # input images
        real_A = self.input_A
        real_B = self.input_B
        self.real_A_encoded = real_A
        self.real_B_encoded = real_B

        # get first cycle
        '''self.real_A_encoded -> self.fake_A_encoded'''
        '''self.real_B_encoded -> self.fake_B_encoded'''
        self.real_A_train = self.myBatchNormlize(self.real_A_encoded)  # real_A_train:norm
        self.fake_A_encoded = self.genA.forward(self.real_A_train)  # fake_A_encoded:norm
        self.fake_B_encoded = self.genB.forward(self.real_B_encoded)  # fake_B_encoded:tensor

        # get perceptual loss
        self.perc_real_A = self.vgg(self.real_A_train).detach()
        self.perc_fake_A = self.vgg(self.fake_A_encoded).detach()

        # get second cycle
        '''self.fake_A_encoded -> self.real_A_recon'''
        '''self.fake_B_encoded -> self.real_B_recon'''
        self.fake_B_encoded = self.myBatchNormlize.forward(self.fake_B_encoded)  # fake_B_encoded:norm
        self.fake_A_tensor = self.myUnnormlize.forward(self.fake_A_encoded)  # fake_A_tensor:tensor
        self.real_B_recon = self.genA.forward(self.fake_B_encoded)  # real_B_recon:norm
        self.real_A_recon = self.genB.forward(self.fake_A_tensor)  # real_A_recon:tensor

        self.real_B_train = self.myBatchNormlize.forward(self.real_B_encoded)  # real_B_train:norm
        self.fake_B_I = self.genA.forward(self.real_B_train)  # fake_B_I:norm

        # self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(), \
        #                                 self.real_A_recon[0:1].detach().cpu(), \
        #                                 self.real_B_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(), \
        #                                 self.real_B_recon[0:1].detach().cpu()), dim=0)

    def update_D(self, opts):
        self.fake_A_encoded = self.fake_A_pool.query(self.fake_A_encoded)  # 50个队列的加载 给判别器使用
        # self.fake_A1 = self.fake_A1_pool.query(self.fake_A1)  # 50个队列的加载 给判别器使用
        self.real_B_recon = self.fake_B_pool.query(self.real_B_recon)  # 50个队列的加载 给判别器使用

        # update disA    判别器优化策略
        self.disA_opt.zero_grad()

        loss_D1_A = self.backward_D_basic(self.disA, self.real_B_train, self.fake_A_encoded)
        loss_D2_A = self.backward_D_basic(self.disA, self.real_B_train, self.real_B_recon)
        self.disA_loss = ((loss_D1_A + loss_D2_A) * 0.5).item()
        self.disA_opt.step()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real1 = self.criterionGAN(pred_real[0], True)
        loss_D_real2 = self.criterionGAN(pred_real[1], True)
        loss_D_real3 = self.criterionGAN(pred_real[2], True)
        loss_D_real = (loss_D_real1 + loss_D_real2 + loss_D_real3) / 3

        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake[0], False)
        loss_D_fake2 = self.criterionGAN(pred_fake[1], False)
        loss_D_fake3 = self.criterionGAN(pred_fake[2], False)
        loss_D_fake = (loss_D_fake1 + loss_D_fake2 + loss_D_fake3) / 3

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def update_EG(self, image_a, image_b, ep, opts):
        self.input_A = image_a
        self.input_B = image_b
        # step——one  判别器以外结构前向传播
        self.forward(ep, opts)
        # step——two  判别器以外结构的优化器梯度置零
        self.genA_opt.zero_grad()
        # step——three 计算判别器以外结构loss
        # step——four  计算判别器以外结构梯度
        self.backward_EG(opts)
        # step——five 判别器以外结构反向优化
        self.genA_opt.step()

    def backward_EG(self, opts):
        # adversarial loss
        disA_out1 = self.disA(self.fake_A_encoded)[0]
        disA_out2 = self.disA(self.real_B_recon)[0]
        loss_G_GAN_A = (self.criterionGAN(disA_out1, True) + self.criterionGAN(disA_out2, True)) * 0.5

        # HBGM
        A = self.real_A_train.clone()
        B = self.fake_A_encoded.clone()
        WR_outA1, WR_outA2 = HBGM(A, B)
        loss_tv = self.ms_ssim_mix(WR_outA1,WR_outA2)*10

        # cross cycle consistency loss
        self.real_A_recon = self.myBatchNormlize(self.real_A_recon)
        loss_G_L1_A = self.TVloss(self.real_A_recon, self.real_A_train) * 100
        loss_G_L1_B = self.ms_ssim_mix(self.real_B_recon, self.real_B_train) * 10

        # perceptual loss
        loss_perceptual = self.criterionL2(self.perc_fake_A, self.perc_real_A) * 0.01

        # Identity loss
        loss_identity_B = self.ms_ssim_mix(self.real_B_train, self.fake_B_I) * 10
        loss_identity = loss_identity_B

        loss_G = loss_G_GAN_A + \
                 loss_G_L1_A + loss_G_L1_B + \
                 loss_identity + \
                 loss_perceptual

        # 计算梯度
        loss_G.backward(retain_graph=True)
        # 损失记录
        self.gan_loss_a = loss_G_GAN_A.item()  # 生成判别
        self.l1_recon_A_loss = loss_G_L1_A.item()  # 循环一致
        self.l1_recon_B_loss = loss_G_L1_B.item()  # 循环一致
        self.perceptual_loss = loss_perceptual.item()  # 感知损失
        self.identity_loss = loss_identity.item()
        self.tvloss = loss_tv.item()
        self.G_loss = loss_G.item()  # 总体损失

    def update_lr(self):
        self.disA_sch.step()
        self.genA_sch.step()

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.disA.load_state_dict(checkpoint['disA'])
        self.genA.load_state_dict(checkpoint['genA'])

        # optimizer
        if train:
            self.disA_opt.load_state_dict(checkpoint['disA_opt'])
            self.genA_opt.load_state_dict(checkpoint['genA_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'disA': self.disA.state_dict(),
            'genA': self.genA.state_dict(),
            'disA_opt': self.disA_opt.state_dict(),
            'genA_opt': self.genA_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def save_dict(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_dict(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A_encoded).detach()
        images_b = self.normalize_image(self.real_B_encoded).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a3 = self.normalize_image(self.real_A_recon).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b3 = self.normalize_image(self.real_B_recon).detach()

        row1 = torch.cat((images_a[0:1, ::], images_a1[0:1, ::], images_a3[0:1, ::]), 3)
        row2 = torch.cat((images_b[0:1, ::], images_b1[0:1, ::], images_b3[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]