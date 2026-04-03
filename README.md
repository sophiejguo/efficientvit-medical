# Accelerating Medical Image Segmentation with EfficientViT-SAM
*Isaac (Zack) Duitz · Sophie Guo · Sam Mitchell · Evan Rubel · Collin Wen*

This is the code repository accompanying the project report [“Accelerating Medical Image Segmentation with EfficientViT-SAM”](Accelerating%20Med%20Img%20Seg%20with%20EfficientVIT-SAM.pdf).

The project compares three models on the CVPR 2024 Segment Anything in Medical Images on Laptop benchmark setting:

- `EfficientViT-SAM`
- `Medficient-SAM`
- `EfficientViT-MedSAM`

The main goal is to understand the tradeoff between segmentation quality and inference efficiency across a diverse medical imaging dataset containing both 2D and 3D cases.

## Background and Motivation

Medical image segmentation is important for diagnosis and treatment planning, but many state-of-the-art promptable segmentation models are too large or too slow for practical deployment in resource-constrained settings.

This project evaluates whether EfficientViT-SAM can provide a better efficiency/accuracy tradeoff for medical segmentation. In particular, it studies:

- zero-shot segmentation performance of EfficientViT-SAM on medical images
- performance of Medficient-SAM as a strong distilled medical baseline
- performance of `EfficientViT-MedSAM`, obtained by fine-tuning EfficientViT-SAM directly on medical segmentation data

The project uses bounding boxes as prompts and evaluates segmentation accuracy and throughput on the CVPR 2024 medical laptop challenge validation set.

## What This Project Studies

The project focuses on three questions:

1. How well does `EfficientViT-SAM` transfer to medical image segmentation in the zero-shot setting?
2. How much does direct medical fine-tuning improve EfficientViT-SAM, producing `EfficientViT-MedSAM`?
3. How does that fine-tuned model compare against `Medficient-SAM`, a stronger medical-domain baseline built with knowledge distillation?

The broader motivation is practical deployment: preserve as much segmentation quality as possible while keeping inference fast enough for large-scale or resource-constrained medical workflows.

## Models

### EfficientViT-SAM

A lightweight promptable segmentation model based on EfficientViT-SAM. It is efficient and general-purpose, but not specialized for medical images.

### Medficient-SAM

A medical segmentation model based on MedSAM with an optimized EfficientViT-style inference pipeline. It uses knowledge distillation from a stronger medical model and serves as the strongest domain-specific baseline in this project.

Reference repo:

- https://github.com/hieplpvip/medficientsam

### EfficientViT-MedSAM

This project’s fine-tuned model. It starts from EfficientViT-SAM and is directly fine-tuned on medical image segmentation data, without the knowledge-distillation pipeline used by Medficient-SAM.

## Dataset

The experiments use the CVPR 2024 Segment Anything in Medical Images on Laptop challenge dataset.

The validation set contains 2D and 3D medical images across 11 modalities:

- CT
- Dermoscopy
- Endoscopy
- Fundus
- Mammography
- Microscopy
- MR
- OCT
- PET
- Ultrasound
- X-Ray


| Modality | Train | Validation |
| --- | ---: | ---: |
| CT | 14622 | 1140 |
| Dermoscopy | 3694 | 66 |
| Endoscopy | 43443 | 200 |
| Fundus | 1057 | 10 |
| Mammography | 1233 | -- |
| Microscopy | 1000 | 50 |
| MR | 5144 | 628 |
| OCT | 1436 | -- |
| PET | 546 | 3 |
| US | 1646 | 600 |
| X-Ray | 34893 | 379 |
| Total | 121718 | 3076 |

Based on the MedficientSAM setup, dataset access requires participating in the challenge and downloading the data separately:

- Challenge: https://www.codabench.org/competitions/1847/

Expected directory layout:

```text
CVPR24-MedSAMLaptopData
├── train_npz
│   ├── CT
│   ├── Dermoscopy
│   ├── Endoscopy
│   ├── Fundus
│   ├── Mammography
│   ├── Microscopy
│   ├── MR
│   ├── OCT
│   ├── PET
│   ├── US
│   └── XRay
└── validation-box
    └── imgs
```

The MedSAM data loader in this repo expects `.npz`-based inputs. The most relevant code is:

- [medsam.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/efficientvit/samcore/data_provider/medsam.py)
- [eval_efficientvit_medsam_model.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/applications/efficientvit_sam/eval_efficientvit_medsam_model.py)

## Method

This project performs two main tasks:

1. Fine-tune EfficientViT-SAM on the CVPR medical training data to produce `EfficientViT-MedSAM`.
2. Evaluate all three models on the validation set using ground-truth bounding boxes as prompts.

Evaluation is performed on both 2D and 3D cases.

For 2D images:

- each mask is predicted from its corresponding bounding box

For 3D images:

- segmentation is performed slice-by-slice using 3D box information
- inference is only run on slices whose z-coordinate lies within at least one bounding box

## Metrics

This project focuses on both segmentation quality and efficiency.

Primary metrics:

- Dice Similarity Coefficient (DSC)
- IoU
- inference throughput for 2D images
- inference throughput for 3D images

The helper script in this repo for evaluation metrics is:

- [calc_acc.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/calc_acc.py)

## Main Results

The main A100 validation-set results are:

| Model | 2D Dice (%) | 2D Throughput (imgs/s) | 3D Dice (%) | 3D Throughput (imgs/s) |
| --- | ---: | ---: | ---: | ---: |
| EfficientViT-SAM | 57.39 | 162.76 | 29.57 | 3.92 |
| EfficientViT-MedSAM | 81.21 | 160.40 | 41.86 | 4.61 |
| Medficient-SAM | 86.42 | 50.00 | 70.21 | 2.33 |

Main conclusions:

- Fine-tuning EfficientViT-SAM on medical data substantially improves accuracy.
- `EfficientViT-MedSAM` keeps nearly the same throughput as vanilla EfficientViT-SAM while improving Dice significantly.
- `Medficient-SAM` achieves the best segmentation accuracy, especially on 3D data, but is slower.
- EfficientViT-based models appear to offer a strong efficiency advantage for medical segmentation workloads.

## EfficientViT-SAM Weights

EfficientViT-SAM weights can be downloaded from:

- https://drive.google.com/drive/folders/1AdpE0s2bFd14BQ4fx-jL5g2TTHiKw2Ky?usp=sharing

The model zoo code expects checkpoints under:

```text
assets/checkpoints/efficientvit_sam/
```

Typical filenames:

- `efficientvit_sam_l0.pt`
- `efficientvit_sam_l1.pt`
- `efficientvit_sam_l2.pt`
- `efficientvit_sam_xl0.pt`
- `efficientvit_sam_xl1.pt`

Some training configs in this repo also reference:

```text
assets/checkpoints/efficientvit_sam/distilled_model/
```

If you keep the config paths unchanged, make sure those checkpoint files exist there, or edit the config to point to your local weights.

## Setup

Basic environment:

```bash
conda create -n efficientvit-medical python=3.10
conda activate efficientvit-medical
pip install -U -r requirements.txt
```

The code assumes:

- PyTorch
- CUDA-capable GPUs
- distributed launch through `torchrun`

## Training

The main medical training entrypoint is:

- [train_efficientvit_medsam_model.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/applications/efficientvit_sam/train_efficientvit_medsam_model.py)

Example:

```bash
torchrun --nproc_per_node=4 applications/efficientvit_sam/train_efficientvit_medsam_model.py \
  applications/efficientvit_sam/configs/efficientvit_sam_l1.yaml \
  --data_provider.root /path/to/CVPR24-MedSAMLaptopData/train_npz \
  --data_provider.dataset medsam \
  --path exp/efficientvit_medsam/efficientvit_medsam_l1
```

Notes:

- [sophie_train.sh](/Users/sophieguo/Documents/GitHub/efficientvit-medical/sophie_train.sh) contains a cluster-specific example.
- The current MedSAM data provider includes a debug-oriented sample cap in the loader implementation.
- `--resume` should only be used if the run directory already contains checkpoints.

## Evaluation / Inference

The main medical evaluation entrypoint is:

- [eval_efficientvit_medsam_model.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/applications/efficientvit_sam/eval_efficientvit_medsam_model.py)

Example:

```bash
torchrun --nproc_per_node=4 applications/efficientvit_sam/eval_efficientvit_medsam_model.py \
  --model efficientvit-sam-l1 \
  --weight_url /path/to/efficientvit_sam_l1.pt \
  --image_size 512 \
  --data_root /path/to/CVPR24-MedSAMLaptopData/validation-box/imgs \
  --output_dir exp/efficientvit_medsam/infer/efficientvit_sam_l1 \
  --save_overlay True
```

This script supports:

- 2D inference
- 3D slice-wise propagation from box prompts
- saving predicted segmentations as compressed `.npz`
- overlay visualization
- prediction time logging

Cluster-specific wrappers:

- [infer.sh](/Users/sophieguo/Documents/GitHub/efficientvit-medical/infer.sh)
- [infer_sam.sh](/Users/sophieguo/Documents/GitHub/efficientvit-medical/infer_sam.sh)

## Additional Local Script

- [main.py](/Users/sophieguo/Documents/GitHub/efficientvit-medical/main.py)

This is a local experimentation script for:

- preprocessing volumetric medical data into slices
- normalizing image data
- running EfficientViT-SAM mask generation on processed slices

It is useful for project experimentation, but it is not the main benchmark training/evaluation entrypoint.

## Repository Structure


- `efficientvit/models`
  EfficientViT backbone and SAM definitions
- `efficientvit/sam_model_zoo.py`
  SAM model factory
- `efficientvit/samcore`
  SAM / MedSAM data providers and trainers
- `applications/efficientvit_sam`
  SAM / MedSAM app scripts




## References

- EfficientViT-SAM paper: Zhang, Zhuoyang, Han Cai, and Song Han. “EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss.”
- MedficientSAM: Le, Bao-Hiep, et al. “MedficientSAM: A robust medical segmentation model with optimized inference pipeline for limited clinical settings.”
- EfficientViT repo: https://github.com/mit-han-lab/efficientvit
- MedficientSAM repo: https://github.com/hieplpvip/medficientsam
