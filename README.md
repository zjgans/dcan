
# Dataset

The following datasets are used in our paper:

 [**miniImageNet**](https://drive.google.com/file/d/12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D/view?usp=sharing) contains 64 classes
for training, 16 classes for validation, and 20 classes for test.
 
 [**tieredImageNet**](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) contains 608 ImageNet classes
that are grouped into 34 high-level categories, which furtherare divided into 20/351, 6/97, and 8/160 categories/classes
for training, validation, and test.
 
 [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing) is derived from CIFAR-100 dataset, which is build by randomly
splitting 100 classes of the CIFAR-100 dataset into 64, 16,and 20 classes for training, validation, and testing, respectively.
 
 [**CUB_200_2011**](https://github.com/zjgans/DeepEMD)  is a fine-grainde dataset
consisting of 11,778 images from 200 bird categories, 100/50/50 classes are divided into train/val/test set.

##  Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test DCAN on the miniImagenet dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/miniimagenet_5w1s.sh
```

##  Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train DCAN on the CUB dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cub_5w1s.sh
```
