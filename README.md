<div align="center">
<img src="assets/beyonddrive_pyrus_nivalis.png" width="800">
<h1>BeyondDrive</h1>
<h4>Beyond Imitation: Learning Safe End-to-End Autonomous Driving from Hard Negatives</h4>

[![Paper](https://img.shields.io/badge/ArXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.19771)
[![License](https://img.shields.io/badge/Apache--2.0-019B8F?style=for-the-badge&logo=apache)](https://github.com/wjl2244/BeyondDrive/blob/main/LICENSE) 

<h4 align="center"><em><a href="https://github.com/wjl2244">Junli Wang</a>, 
<a href="https://github.com/Zhihua-Hua">Zhihua Hua</a>,
<a href="https://github.com/Liuxueyi">Xueyi Liu</a>, 
<a href="https://github.com/ZebinX">Zebing Xing</a>,  
<a href="https://github.com/hctian713">Haochen Tian</a>, 
<a href="https://openreview.net/profile?id=~Kun_Ma3">Kun Ma</a>, 
<a href="https://openreview.net/profile?id=%7EGuang_Chen1">Guang Chen</a>, 
<a href="https://scholar.google.com/citations?user=68tXhe8AAAAJ&hl=en">Hangjun Ye</a>, 
<a href="https://long.ooo/">Long Chen</a>, 
<a href="https://scholar.google.com/citations?user=snkECPAAAAAJ&hl=en">Qichao Zhang</a>📧</em>
</h4>

<h4 align="center">
<br>📧 indicates corresponding authors.<br>
<b > CASIA &nbsp; | &nbsp; Xiaomi Embodied Intelligence Team  &nbsp; | &nbsp; Fudan University </b>
</h4>
</div>

---

## 📢 News
- **`[2026/5/20]`** We released code and checkpoints.
- **`[2026/5/20]`** We released our [paper](https://arxiv.org/abs/2602.20060) on arXiv. 

## 📌 Table of Contents
- 🏛️ [Model Zoo](#%EF%B8%8F-model-zoo)
- 🎯 [Getting Started](#-getting-started)
- 📦 [Data Preparation](#-data-preparation)
  - [Download Dataset](#1-download-dataset)
  - [Set Up Configuration](#2-set-up-configuration)
  - [Cache the Dataset](#3-cache-the-dataset)
- ⚙️ [Training and Evaluation](#%EF%B8%8F-training-and-evaluation)
  - [Evaluation](#1-evaluation)
  - [Training](#2-training)
- ❤️ [Acknowledgements](#%EF%B8%8F-acknowledgements)

## 🏛️ Model Zoo

| Method | Backbone | Benchmark | PDMS | Weight Download |
| :---: | :---: | :---:  | :---:  | :---: |
| FlowPolicy | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 87.0 | [Google Drive](https://drive.google.com/file/d/1wrUrRAQoRPL6XLgZtgBA72SWGMq1kOTr/view?usp=drive_link) |
| LTF* | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 88.7 | Google Drive |
| LTFv7 | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 89.8 | [Google Drive](https://drive.google.com/file/d/1ctVDgZf5yAr0oHiUKUe0CSOWWv3fRdZA/view?usp=drive_link) |
| MeanFuser + BeyondDrive | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 90.3 | [Google Drive](https://drive.google.com/file/d/1yXXlCLWfywht1S1wGliNow5e56_0-9YN/view?usp=drive_link) |
| DiffusionDrive + BeyondDrive | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 89.2 | [Google Drive](https://drive.google.com/file/d/1J7JYhrs0enih8XslHOr4F43bSHT7yr0Z/view?usp=drive_link) |
| WoTE + BeyondDrive | [ResNet-34](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link) | NAVSIM | 89.2 | [Google Drive](https://drive.google.com/file/d/17p77Q3jhWqyUCSaUGOp9yVV52PpO-AW3/view?usp=drive_link) |


## 🎯 Getting Started

### 1. Clone MeanFuser Repo

```bash
git clone https://github.com/wjl2244/BeyondDrive.git
cd BeyondDrive
```

### 2. Create Environment

```bash
conda create -n beyonddrive python=3.9 -y
conda activate beyonddrive
pip install -e .
```

## 📦 Data Preparation
**NOTE: Please review and agree to the [LICENSE file](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) file before downloading the data.**

### 1. Download Dataset

#### a. Download via NAVSIM offical installation.
Follow the instructions in the [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md#2-download-the-dataset) to download the dataset.


#### b. Download via Hugging Face
Alternatively, you can download the dataset using Hugging Face with the following commands:
```bash
export HF_ENDPOINT="https://huggingface.co"
# export HF_ENDPOINT="http://hf-mirror.com"  # Uncomment this line if you are in China

# Install the huggingface_hub tool
pip install -U "huggingface_hub"

# Download the OpenScene dataset
hf download --repo-type dataset OpenDriveLab/OpenScene --local-dir ./navsim_dataset/ --include "openscene-v1.1/*"

# Download the map data
cd download && ./download_maps.sh
```

### 2. Set Up Configuration
Move the download data to create the following structure.

```angular2html
navsim_workspace/
├── MeanFuser/
├── dataset/
│    ├── maps/
│    ├── navsim_logs/
│    │   ├── test/
│    │   ├── trainval/
│    ├── sensor_blobs/
│    │   ├── test/
│    │   ├── trainval/
└── cache/
     ├── navtest_v1_metric_cache/
     └── traintest_v1_cache/
```

### 3. Cache the Dataset
We provide a script to cache the dataset and metrics.
```bash
cd BeyondDrive

# Cache the dataset. (navtrain and navtest)
bash scripts/evaluation/run_dataset_cache.sh

# Cache the metric.
bash scripts/evaluation/run_metric_cache.sh
```

## ⚙️ Training and Evaluation

### 1. Evaluation
(1) Please download the pre-trained checkpoints from [here](https://drive.google.com/drive/folders/1VGzTzvoJkd65aGLn5bp64r86QLrcPxI3?usp=sharing) and place them in the `navsim_workspace/BeyondDrive/exp/` directory.

(2) Please download the ResNet-34 pretrained weights from [here](https://drive.google.com/file/d/1-6mtwHsrZt4TyH4lfFEJTT8_dnnkejAI/view?usp=drive_link). After downloading, update the corresponding path in the configuration file:
`navsim/agents/transfuser/transfuser_config.py`,
`navsim/agents/flowpolicy/flowpolicy_config.py`,
`navsim/agents/meanfuser/meanfuser_config.py`,
`navsim/agents/diffusiondrive/diffusiondrive_config.py`

```bash
cd BeyondDrive

# FlowPolicy Evaluation
bash scripts/evaluation/run_evaluation_flowpolicy.sh

# Negative Samples Generation and Evaluation
bash scripts/evaluation/run_generate_negative_samples_pool.sh

# LTF* & LTFv7
bash scripts/evaluation/run_evaluation_transfuser.sh

# MeanFuser+BeyondDrive
bash scripts/evaluation/run_evaluation_meanfuser.sh

# DiffusionDrive+BeyondDrive
bash scripts/evaluation/run_evaluation_diffusiondrive.sh
```

### 2. Training

```bash
cd BeyondDrive

# FlowPolicy Training
bash scripts/training/run_training_flowpolicy.sh

# LTF* & LTFv7
bash scripts/training/run_training_transfuser.sh

# MeanFuser+BeyondDrive
bash scripts/training/run_training_meanfuser.sh

# DiffusionDrive+BeyondDrive
bash scripts/training/run_training_diffusiondrive.sh
```

## ❤️ Acknowledgements

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [MeanFlow](https://github.com/zhuyu-cs/MeanFlow) | [NAVSIM](https://github.com/autonomousvision/navsim) | [HUGSIM](https://github.com/hyzhou404/NAVSIM) | [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) | [GTRS](https://github.com/NVlabs/GTRS)
