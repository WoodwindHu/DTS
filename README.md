# DTS: Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection
DTS is an unsupervised domain adaption method on 3D object detection, which is accepted on CVPR2023.

> [**Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection**](https://arxiv.org/abs/2304.09446)         
> Qianjiang Hu, Daizong Liu, Wei Hu 

Copyright (C) 2023 Qianjiang Hu, Daizong Liu, Wei Hu

License: MIT for academic use.

Contact: Wei Hu (forhuwei@pku.edu.cn)

# Introduction

3D object detection from point clouds is crucial in safety-critical autonomous driving.
Although many works have made great efforts and achieved significant progress on this task, most of them suffer from expensive annotation cost and poor transferability to unknown data due to the domain gap.
Recently, few works attempt to tackle the domain gap in objects, but still fail to adapt to the gap of varying beam-densities between two domains, which is critical to mitigate the characteristic differences of the LiDAR collectors.
To this end, we make the attempt to propose a density-insensitive domain adaption framework to address the density-induced domain gap.
In particular, we first introduce Random Beam Re-Sampling (RBRS) to enhance the robustness of 3D detectors trained on the source domain to the varying beam-density.
Then, we take this pre-trained detector as the backbone model, and feed the unlabeled target domain data into our newly designed task-specific teacher-student framework for predicting its high-quality pseudo labels.
To further adapt the property of density-insensitive into the target domain, we feed the teacher and student branches with the same sample of different densities, and propose an Object Graph Alignment (OGA) module to construct two object-graphs between the two branches for enforcing the consistency in both the attribute and relation of cross-density objects.
Experimental results on three widely adopted 3D object detection datasets demonstrate that our proposed domain adaption method outperforms the state-of-the-art methods, especially over varying-density data.

# Model Zoo

## Model Zoo

### nuScenes -> KITTI TASK
|                                                                                                | Car@R40 |  download |
|------------------------------------------------------------------------------------------------|:-------:|:---------:|
| [SECOND-IoU](tools/cfgs/da-nuscenes-kitti_models/secondiou_dts/dts.yaml)                       |  66.6   | [model](https://drive.google.com/file/d/12YFee846dvCgyEeiaVGke-x6I8sNjjiM/view?usp=share_link) | 
| [PV-RCNN](tools/cfgs/da-nuscenes-kitti_models/pvrcnn_dts/dts.yaml)                             |  71.8   | [model](https://drive.google.com/file/d/1CBjimQMctengq58ZEg2dadu2H85qsyl4/view?usp=share_link) |
| [PointPillar](tools/cfgs/da-nuscenes-kitti_models/pointpillars_dts/dts.yaml)                   |  51.8   | [model](https://drive.google.com/file/d/1vRd99-s_q-rM07d7_0ft_KI68ON3i94-/view?usp=share_link)  | 


### Waymo -> KITTI TASK

|                                                                                                 | Car@R40 |   download  | 
|-------------------------------------------------------------------------------------------------|:-------:|:-----------:|
| [SECOND-IoU](tools/cfgs/da-waymo-kitti_models/secondiou_dts/dts.yaml)                           |  71.5  |  [model](https://drive.google.com/file/d/1KiMt-IkYlOKOABo2a1O3VJWOnaJKdlyK/view?usp=share_link)  | 
| [PVRCNN](tools/cfgs/da-waymo-kitti_models/pvrcnn_dts/dts.yaml)                                  |  68.1  |  [model](https://drive.google.com/file/d/13GYGRl_KZ42nJur1mCiwAq1yh2xpzvka/view?usp=share_link)  | 
| [PointPillar](tools/cfgs/da-waymo-kitti_models/pointpillars_dts/dts.yaml)                       |  50.2  |  [model](https://drive.google.com/file/d/1L-bCrulR0WJrX-wYSfWarn0rLMAmGENo/view?usp=share_link)  | 


### Waymo -> nuScenes TASK
|                                                                         | Car@R40 | download | 
|-------------------------------------------------------------------------|:-------:|:--------:|
| [SECOND-IoU](tools/cfgs/da-waymo-nus_models/secondiou_dts/dts.yaml)     | 23.0   | [model](https://drive.google.com/file/d/1uxUowfkCWu9rdgANTAAnWaH6ahGNa8Z4/view?usp=share_link) | 
| [PVRCNN](tools/cfgs/da-waymo-nus_models/pvrcnn_dts/dts.yaml)            | 26.2   | [model](https://drive.google.com/file/d/1heb7q3D3OVq-Mu5xvz8OpfjZ4kLm9eOH/view?usp=share_link) |
| [PointPillar](tools/cfgs/da-waymo-nus_models/pointpillars_dts/dts.yaml) | 21.5   | [model](https://drive.google.com/file/d/1Oa8ZF35-mZRV9bAGmnBI4XtP4l9RveId/view?usp=share_link)  | 


# Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

# Usage

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

# Citation
```
@inproceedings{hu2023density,
  title={Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection},
  author={Hu, Qianjiang and Liu, Daizong and Hu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
