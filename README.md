# DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping

Shiqi Yang, Hanlin Qin, Shuai Yuan, Xiang Yan, Hossein Rahmani, IEEE Transactions on Instrumentation and Measurement, 2024 [[Paper]](https://ieeexplore.ieee.org/document/10711892) [[Weight]](https://drive.google.com/file/d/1VhCR8dTqmqpQSaA_f4GokRzaCCXq5cG8/view?usp=sharing)


# Chanlleges and inspiration   
![Image text](https://github.com/xdFai/DestripeCycleGAN/blob/main/Fig/image0.png)


# Structure
![Image text](https://github.com/xdFai/DestripeCycleGAN/blob/main/Fig/image1.png)

![Image text](https://github.com/xdFai/DestripeCycleGAN/blob/main/Fig/image2.png)


# Introduction
DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping. Shiqi Yang, Hanlin Qin, Shuai Yuan, Xiang Yan


The main contributions of this paper are as follows: 
1. An efficient deep unsupervised DestripeCycleGAN is proposed for infrared image destriping. We incorporated a stripe generation model (SGM) into the framework, balancing the semantic information between the degraded and clean domains.

2. The Haar Wavelet Background Guidance Module (HBGM) is designed to mitigate the impact of vertical stripes and accurately assess the consistency of background details. As a plug-and-play image constraint module, it can offer a powerful unsupervised restriction for DestripeCycleGAN.
   
3. We design multi-level wavelet U-Net (MWUNet) that leverages Haar wavelet sampling to minimize feature loss. The network effectively integrates multi-scale features and strengthens long-range dependencies by using group fusion block (GFB) in skip connections.


## Citation

If you find the code useful, please consider citing our paper using the following BibTeX entry.

```
@ARTICLE{10711892,
  author={Yang, Shiqi and Qin, Hanlin and Yuan, Shuai and Yan, Xiang and Rahmani, Hossein},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping}, 
  year={2024},
  volume={73},
  number={},
  pages={1-14},
  keywords={Noise;Wavelet transforms;Generators;Semantics;Noise reduction;Image restoration;Computational modeling;Adaptation models;Wavelet domain;Image reconstruction;Convolutional neural network (CNN);CycleGAN;infrared image destriping;stripe prior modeling;unsupervised learning},
  doi={10.1109/TIM.2024.3476560}}

```

## Usage


#### 1. Data
* **Our project has the following structure:**

##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.  [[Weight]](https://drive.google.com/file/d/1VhCR8dTqmqpQSaA_f4GokRzaCCXq5cG8/view?usp=sharing)
```bash
python test.py
```

## Contact
**Welcome to raise issues or email to [22191214967@stu.xidian.edu.cn](22191214967@stu.xidian.edu.cn) or [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) for any question regarding our DestripeCycleGAN.**
