import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--valphase', type=str, default='val', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=64, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=64, help='cropped image size for training')
        self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
        self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')
        self.parser.add_argument('--no_flip', action='store_true', default=False, help='specified if no flipping')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='7_3_test', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='../results',
                                 help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=20, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

        # training related
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', action='store_true',
                                 help='use spectral normalization in discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='warmup', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=150, help='number of epochs')
        self.parser.add_argument('--n_ep_decay', type=int, default=100,
                                 help='epoch start decay learning rate, set -1 if no decay')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--train_path', type=str, default=r'/media/omnisky/c91e9985-5113-463d-83a6-6ec3405ef3a7/ysq/SCI02/datasets/new_jiaozheng', help='path of training data')
        self.parser.add_argument('--val_path', type=str, default=r'/media/omnisky/c91e9985-5113-463d-83a6-6ec3405ef3a7/ysq/SCI02/datasets/new_jiaozheng', help='path of testing data')
        self.parser.add_argument('--a2b', type=int, default=0, help='translation direction, 1 for a2b, 0 for b2a')
        self.parser.add_argument('--gan_mode', type=str, default='vanilla',
                                 help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
        self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
        self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
        self.parser.add_argument('--a2b', type=int, default=0, help='translation direction, 1 for a2b, 0 for b2a')

        # ouptput related
        self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
        self.parser.add_argument('--name', type=str, default=r'', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')

        # model related
        self.parser.add_argument('--resume', type=str, default='', help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--test_path', type=str, default='', help='path of testing data')
        self.parser.add_argument('--val_path', type=str, default=r'', help='path of testing data')
        self.parser.add_argument('--valphase', type=str, default='val', help='phase for dataloading')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan',
                                 help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt
