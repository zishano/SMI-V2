# [] Sparse Model Inversion V2: Towards Efficient and Accurate Inversion of Vision Transformers for Data-Free Compression

## Abstract
Vision transformers (ViTs)  have achieved significant success in various computer vision tasks, enabling their deployment on privacy-protected devices. These devices generally have privacy protection requirements, which motivates the development of model inversion techniques such as Sparse Model Inversion (SMI) specifically designed for ViTs.
SMI is a promising approach for model inversion, enabling the reconstruction of original data in data-free compression scenarios. Although SMI improves inversion efficiency via background patch pruning, it provides limited treatment of the accuracy dimension. Therefore, for privacy-protected devices, it is crucial to develop accurate and efficient model inversion strategies. 
In this paper, we propose a two-pronged approach, SMI V2, for efficient and accurate model inversion toward data-free compression of ViTs.
Firstly, in terms of efficiency, we identify the redundancy in current inversion techniques by analyzing the token value similarity index. We introduce a token fusion strategy, specifically designed for the inversion setting, that identifies and merges the most semantically similar tokens, thereby reducing the computational complexity of image inversion by decreasing the number of tokens involved. 
Secondly, for accuracy, we clarify the uncertainty of the inverted image label by measuring the label discrepancy and analyzing the semantic boundary. We then introduce a label compensation function to capture the ground-truth labels of the images accurately.
We establish data-free model compression benchmarks, including quantization, sparsity, and distillation. Extensive experimental results demonstrate that SMI V2 significantly accelerates the inversion process through the token fusion strategy and label compensation function while maintaining or enhancing model performance in data-free compression of ViTs.


# Requirements
One high-end GPU for inference such as an RTX 3090
* Install [PyTorch](http://pytorch.org/)
* pip install -r requirements.txt

# Model Quantization
  - Example: Quantize (W8/A8) DeiT/16-Base with inverted data.
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

@inproceedings{li2025eff,
  title={Eff-DFQT: Efficient Model Inversion for Data-free Quantization of Vision Transformers},
  author={Li, Mengkui and Chen, Xinrui and Chen, Hai and Zhao, Kang and Zhang, Yanping and Zhao, Shu and Qian, Fulan},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```