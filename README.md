# DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping

Shiqi Yang, Hanlin Qin, Shuai Yuan, Xiang Yan, Hossein Rahmani, IEEE Transactions on Instrumentation and Measurement, 2024 [[Paper]](https://arxiv.org/abs/2402.09101)

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
@article{yang2024destripecyclegan,
  title={DestripeCycleGAN: Stripe Simulation CycleGAN for Unsupervised Infrared Image Destriping},
  author={Yang, Shiqi and Qin, Hanlin and Yuan, Shuai and Yan, Xiang and Rahmani, Hossein},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2024}
}
```

## Usage


#### 1. Data
* **Our project has the following structure:**

##### 2. Train.
```bash
python train.py
```

#### 3. Test and demo.
```bash
python test.py
```

## Contact
**Welcome to raise issues or email to [22191214967@stu.xidian.edu.cn](22191214967@stu.xidian.edu.cn) or [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) for any question regarding our DestripeCycleGAN.**
