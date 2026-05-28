# [ICME 2025] Eff-DFQT: Efficient Model Inversion for Data-free Quantization of Vision Transformers

## Abstract
Model inversion is a promising technique for raw data reconstruction, especially in data-free quantization of Vision Transformers (ViTs). Previous inversion methods for ViTs have focused on extracting necessary foreground information while discarding irrelevant noise. However, these mode inversion methods for ViTs are inefficient in terms of data synthesis speed. In this paper, we propose a novel method to accelerate model inversion for efficient data-free quantization of ViTs(Eff-DFQT). Our method has the following features. 1) Token fusion strategy tailored for model inversion. We propose a token fusion strategy tailored for model inversion to lower the computations for image inversion. 2) Label compensation function. We propose a label compensation function to model the label uncertainty of the inverted image and accurately capture the real labels, which improves the quality of the inverted data by compensating for the negative effects of token reduction. Extensive experimental results demonstrate that Eff-DFQT, significantly accelerates the inversion process through token fusion strategy tailored for model inversion and label compensation function, while maintaining or even improving model performance in data-free quantization of ViTs.

# Eff-DFQT
We plan to make the source code related to this paper public in the near future. Please continue to follow this project for the latest information and updates. Thank you for your patience and support!
![img10](https://github.com/user-attachments/assets/5500b323-5a2d-4ac6-bfa7-451ef45b8d2b)

# Requirements
One high-end GPU for inference such as an RTX 3090
* Install [PyTorch](http://pytorch.org/)
* pip install -r requirements.txt

# Model Quantization
  - Example: Quantize (W8/A8) DeiT/16-Base with inverted data (Eff-DFQT).
```
python test_quant_tome_test.py --model deit_base_16_imagenet \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --dataset ../data/imagenet \
    --datapool ./output \
    --mode 0 \
    --w_bit 8 --a_bit 8 \
    --calib-batchsize 128 \
    --val-batchsize 50 \
    --gpu 0 \
    --ldr 0 \
    --ratio 1 1 1 1 1 1 1 1 1 1 1 1
```

## BibTex
```bash
@inproceedings{li2025eff,
  title={Eff-DFQT: Efficient Model Inversion for Data-free Quantization of Vision Transformers},
  author={Li, Mengkui and Chen, Xinrui and Chen, Hai and Zhao, Kang and Zhang, Yanping and Zhao, Shu and Qian, Fulan},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```

## Acknowledge
```bash
@inproceedings{
hu2024sparse,
title={Sparse Model Inversion: Efficient Inversion of Vision Transformers for Data-Free Applications},
author={Zixuan Hu and Yongxian Wei and Li Shen and Zhenyi Wang and Lei Li and Chun Yuan and Dacheng Tao},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=T0lFfO8HaK}
}
```