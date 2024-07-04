# coding=utf-8
import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, Pad, ToPILImage
import random
import cv2

class dataset_single_test(data.Dataset):
  def __init__(self, opts, input_dim):
    self.test_path = opts.test_path
    images = os.listdir(os.path.join(self.test_path, '00042_set12_0.1'))

    images.sort()
    self.img = [os.path.join(self.test_path,'00042_set12_0.1', x) for x in images]

    self.size = len(self.img)
    self.input_dim = input_dim
    self.img_name = self.img
    transforms1 = [Pad((0, 0, 3, 3), padding_mode='edge'), ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms1 = Compose(transforms1)
    transforms2 = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    self.transforms2 = Compose(transforms2)
    return

  def __getitem__(self, index):
    data, needcrop, img_names = self.load_img(self.img[index], self.input_dim)
    return data, needcrop, img_names

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    y = cv2.imread(img_name)
    h,w = y.shape[0], y.shape[1]
    needcrop = 0
    if h == 321 or w == 321:
      img = self.transforms1(img)
      needcrop = 1
    else:
      img = self.transforms2(img)    
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    img_names = [os.path.splitext(x.split('/')[-1])[0] for x in self.img_name]
    return img, needcrop, img_names

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.train_path = opts.train_path
    # A
    images_A = os.listdir(os.path.join(self.train_path, opts.phase + 'A'))
    self.A = [os.path.join(self.train_path, opts.phase + 'A', x) for x in images_A]
    # B
    images_B = os.listdir(os.path.join(self.train_path, opts.phase + 'B'))
    self.B = [os.path.join(self.train_path, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    transforms = [ToTensor()]
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class dataset_unpair_val(data.Dataset):
  def __init__(self, opts):
    self.val_path = opts.val_path

    # A
    images_A = os.listdir(os.path.join(self.val_path, opts.valphase + 'A_1007'))
    self.A = [os.path.join(self.val_path, opts.valphase + 'A_1007', x) for x in images_A]
    # B
    images_B = os.listdir(os.path.join(self.val_path, opts.valphase + 'B_1007'))
    self.B = [os.path.join(self.val_path, opts.valphase + 'B_1007', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    # transforms = [ToTensor()]
    transforms = [ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    transforms_clean = [ToTensor()]
    self.transforms = Compose(transforms)
    self.transforms_clean = Compose(transforms_clean)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img_clean(self.B[index], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img_clean(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def load_img_clean(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms_clean(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size