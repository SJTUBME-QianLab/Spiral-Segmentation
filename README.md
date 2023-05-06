
This repository holds the PyTorch code for the paper

**Model-driven deep learning method for pancreatic cancer segmentation based on spiral-transformation** 
 

All the materials released in this library can ONLY be used for RESEARCH purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserve the copyright and all legal rights of these codes.

# Author List
X Chen, Z Chen, J Li, YD Zhang, X Lin, Xiaohua Qian*

# Abstract
Pancreatic cancer is a lethal malignant tumor with one of the worst prognoses. Accurate segmentation of pancreatic cancer is vital in both clinical diagnosis and treatment. Due to the unclear boundary and small size tumors, it is a challenge for both manually annotate and automatically segment tumor regions. According to the dilemma of the 3D information utilization and small sample sizes, we proposed a model-driven deep learning model for automated pancreatic cancer segmentation based on the spiral-transformation. Specifically, a novel spiral-transformation method with uniform sampling was developed for the segmentation model to map the 3D imaging to a 2D plane while preserving spatial relationship of texture, addressing the challenge of effectively applying 3D contextual information in the 2D model. The spiral-transformation is also the first time to be introduced in the segmentation model to provide an effective data augmentation solution for alleviating the issue of small sample size. Besides, the 3D rebuilding processing was embedded into the deep learning model to unify a whole 2D segmentation framework for overcoming the problem of non-unique 3D rebuilding results caused by the uniform and dense sampling in the spiral-transformation. A smooth regularization based on the rebuilding prior knowledge was also designed to optimize the segmentation results. The extensive experiments show that the proposed model achieved a promising segmentation performance on multi-parameter MRIs (such as T2w and T1w) with the accuracies more than 64%, and outperformed the state-of-the-art methods in the public pancreas cancer CT dataset. This combined spiral-transformation and model-driven 2D deep learning segmentation scheme can also provide a novel paradigm for medical image analysis community to deal with the tumor segmentation.

# quickly train and test
python run_python.py

# Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{
  title     = {Model-driven deep learning method for pancreatic cancer segmentation based on spiral-transformation},
  author    = {X Chen, Z Chen, J Li, YD Zhang, X Lin, Xiaohua Qian*},
  journal   = {IEEE Transactions on Medical Imaging},
  month     = {January}，
  year      = {2022},
}
```
