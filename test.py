import torch
from options import TestOptions
from dataset import dataset_single_test
from model_singalG import DerainCycleGAN

import os
import time
import torchvision
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def normalization(data):
  _range = np.max(data) - np.min(data)
  return (data - np.min(data)) / _range

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  dataset = dataset_single_test(opts, opts.input_dim_a)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DerainCycleGAN(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  time_test = 0
  count = 0
  for idx1, (img1, needcrop, img_names) in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1 = img1.cuda()
    imgs = []
    masks = []
    names = []
    start_time = time.time()
    # for idx2 in range(1):
    with torch.no_grad():
      # img, mask = model.test_forward(img1, a2b=opts.a2b)
      img = model.test_forward(img1, a2b=opts.a2b)

    img = torch.clamp(img, -1., 1.)
    img = (img + 1) / 2
    end_time = time.time()
    dur_time = end_time - start_time
    time_test += dur_time
    print(img_names[idx1], ': ', dur_time)
    imgs.append(img)
    torchvision.utils.save_image(img, os.path.join(result_dir, (img_names[idx1][0] + '.png')), nrow=1)
    count += 1
  print('Avg. time:%.4f' % (time_test / count))

  return

if __name__ == '__main__':
  main()
