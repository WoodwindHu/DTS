# DTS
DTA is an unsupervised domain adaption method on 3D object detection, which is accepted on CVPR2021.

Copyright (C) 2020 Qianjiang Hu*, Xiao Wang*, Wei Hu, Guo-Jun Qi

License: MIT for academic use.

Contact: Guo-Jun Qi (guojunq@gmail.com)

# Introduction

3D object detection from point clouds is crucial in safety-critical autonomous driving.
Although many works have made great efforts and achieved significant progress on this task, most of them suffer from expensive annotation cost and poor transferability to unknown data due to the domain gap.
Recently, few works attempt to tackle the domain gap in objects, but still fail to adapt to the gap of varying beam-densities between two domains, which is critical to mitigate the characteristic differences of the LiDAR collectors.
To this end, we make the attempt to propose a density-insensitive domain adaption framework to address the density-induced domain gap.
In particular, we first introduce Random Beam Re-Sampling (RBRS) to enhance the robustness of 3D detectors trained on the source domain to the varying beam-density.
Then, we take this pre-trained detector as the backbone model, and feed the unlabeled target domain data into our newly designed task-specific teacher-student framework for predicting its high-quality pseudo labels.
To further adapt the property of density-insensitive into the target domain, we feed the teacher and student branches with the same sample of different densities, and propose an Object Graph Alignment (OGA) module to construct two object-graphs between the two branches for enforcing the consistency in both the attribute and relation of cross-density objects.
Experimental results on three widely adopted 3D object detection datasets demonstrate that our proposed domain adaption method outperforms the state-of-the-art methods, especially over varying-density data.

# Installation
Coming soon

# Usage
Coming soon

# Citation
```
@inproceedings{hu2023density,
  title={Density-Insensitive Unsupervised Domain Adaption on 3D Object Detection},
  author={Hu, Qianjiang and Liu, Daizong and Hu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
