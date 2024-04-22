# Recognizing Deviant Place
Identifying and mapping urban Crime-Hotspots using a deep-learning model and sequential images of deviant locations (i.e., Location of reported deviant incidents such as crime and civil complaints).
#### [<a href="https://aaai.org/Conferences/AAAI-22/">AAAI2022</a>] "DevianceNet: Learning to Predict Deviance from A Large-scale Geo-tagged Dataset"
#### [TPAMI2024] "What Makes Deviant Places?"
- This repository contains the code for the DevianceNet applied to the deviance prediction.

# Video
https://github.com/JinhwiPark/DevianceNet/assets/70624468/9ad55a68-1b19-4800-8255-c4f66a3f7d66


# Datatset
### Datatset Examples
<img src = "https://github.com/JinhwiPark/DevianceNet/assets/70624468/5f23c5dd-6156-4801-b1eb-3477ca12674b" width="40%" height="40%">

### Datatset Composition
 - Sequential images
 - Geo-tagged image files annotated with deviance class
 - Dataset description


### Dataset URL 
Please download the dataset at (https://drive.google.com/drive/folders/1ERFaC_6IseQgXDvs5_ep5at56b-b7Vmg?usp=sharing).


### Dataset Description
```
Deviance
├── Seoul
│    ├── Seoul_train_SEA 
│    │    ├── v_class1_g37.499,127.1251_c0
│    │    │     ├── frame000000.jpg
│    │    │     └── ...
│    │    └── ...
│    ├── Seoul_train_DIA
│    │    └── ...
│    ├── Seoul_test_SEA
│    │    └── ...
│    └── Seoul_test_DIA
│         └── ...
├── Busan
│    └── ...    
├── Incheon
│    └── ...    
├── Daejeon
│    └── ...    
├── Daegu
│    └── ...    
├── Newyork
│    └── ...    
└── Chicago
     └── ...   
```

The directory {City}\_{Train/Test}\_{SEA/DIA} is used for train/test corresponding to the {City}.

The files v\_class{Number}\_g{GPS}\_c{Direction} is annotated with corresponding deviance class, GPS and viewpoint direction.

Note that '+'(ex. v_class2_g35.0911,129.0394_c1+) indicates the additional sets of sequential images in the GPS. 


# Code
## The source code contains
 - Our implementation of DevianeNet (./models/devianceNet.py)
 - Official implementation of DevianceNet (./main.py)
 - Train & evaluation code for Deviance Dataset
 - Code description

## Requirements
 - Ubuntu 16.04
 - python 3.8
 - numpy>=1.18.5
 - torch==1.7.0
 - torchvision>=0.5.0
 - pillow>= 8.0.1
 - scikit-image
 - tqdm
 - sklearn
 - pandas
 - h5py
 - matplotlib
 - apex
 - scipy>=1.4.1

### Preparation
1. Download the dataset and pretrained weights at (https://drive.google.com/drive/folders/1ERFaC_6IseQgXDvs5_ep5at56b-b7Vmg?usp=sharing).
2. Put the dataset into **./data/** folder
3. [Application & Evaluation] Put the pretrained weights into **./weight_file/** folder.

### Training & Testing
Use the **'main.py'** to train/test our model.
```shell
# Train
python main.py --train_folder_directory {Dataset Directory} --SEA_folder_directory {Dataset Directory} --DIA_folder_directory {Dataset Directory} --experiment_description Train --batch_size 36 --num_threads 16 --classifier_type SEA_DIA
# Test SEA
python main.py --SEA_folder_directory {Dataset Directory} --experiment_description Test_SEA --batch_size 1 --classifier_type SEA_DIA --test_only --weight_load_pth ./weight_file/[AAAI2022-DevianceNet]SEA.pth --test_metric SEA
# Test DIA
python main.py --DIA_folder_directory {Dataset Directory} --experiment_description Test_DIA --batch_size 1 --classifier_type SEA_DIA --test_only --weight_load_pth ./weight_file/[AAAI2022-DevianceNet]DIA.pth --test_metric DIA
# Application
python main.py --SEA_folder_directory ./application --experiment_description Test_DIA --batch_size 1 --classifier_type SEA_DIA --test_only --weight_load_pth ./weight_file/[AAAI2022-DevianceNet]SEA.pth --test_metric SEA
```

## Citation
If you find this code useful for your research, please cite our paper :)

```bibtex
@inproceedings{park2022deviance,
  title={DevianceNet: Learning to Predict Deviance from A Large-scale Geo-tagged Dataset},
  author={Park, Jin-Hwi and Park, Young-Jae and Lee, Junoh and Jeon, Hae-Gon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```


### Acknowledgement	
Part of our code is borrowed from following implementations. We thank the authors for releasing their code and models.
- DELTAS: Depth Estimation by Learning Triangulation And densification of Sparse points (ECCV 2020) [Code](https://github.com/magicleap/DELTAS) [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660103.pdf)
- A Closer Look at Spatiotemporal Convolutions for Action Recognition (CVPR 2018)
[Code](https://github.com/irhum/R2Plus1D-PyTorch) [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tran_A_Closer_Look_CVPR_2018_paper.pdf)

