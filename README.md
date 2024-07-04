# This is the code of  paper 'DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping'

[[Paper]](https://arxiv.org/abs/2402.09101)

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
