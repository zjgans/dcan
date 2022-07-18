# Learning Dynamic Concept-Aware Network for Few-Shot Learning
# Dataset

The following datasets are used in our paper:

 [**miniImageNet**](https://drive.google.com/file/d/12V7qi-AjrYi6OoJdYcN_k502BM_jcP8D/view?usp=sharing) 
 
 [**tieredImageNet**](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)
 
 [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

## :pushpin: Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test DCAN on the miniImagenet dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/miniimagenet_5w1s.sh
```

## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train DCAN on the CUB dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cub_5w1s.sh
```
