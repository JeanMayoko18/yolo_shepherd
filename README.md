# SHEPHERD: Real-Time Detection of Sheep and Predators for Autonomous Herding Robots

## 🐑 Overview
**SHEPHERD** is a lightweight and efficient object detection model tailored for autonomous herding scenarios. Built upon YOLOv8n, it integrates advanced components such as **ConvNeXtBlockV2**, **SE Attention**, and **Wise-IoU Loss**, achieving superior performance in detecting sheep and predators (wolves, dogs, etc.) in real-time field conditions.

The model is optimized for deployment on mobile robots (e.g., Unitree Go1) using **ROS 2 Humble**.

## 🧠 Key Model Features

- ⚙️ **Architecture**: Based on YOLOv8n with ConvNeXtBlockV2 and SE Attention
- 🧮 **Loss Function**: Wise-IoU for enhanced localization
- 📊 **Performance**:
  - **mAP@0.5 (Val):** 72.0%
  - **mAP@0.5 (Test):** 57.1%
  - **Precision/Recall:** 0.75 / 0.74
  - **FPS:** ~293 (on RTX 4070 Ti SUPER)
  - **Model Size:** 6.29 MB

## 📦 Dataset: Herding Detection Dataset

This dataset supports training and evaluation of the SHEPHERD model. It includes real-world images and labels of five classes:

| ID | Class  |
|----|--------|
| 0  | person |
| 1  | dog    |
| 2  | horse  |
| 3  | sheep  |
| 4  | wolf   |

### 🔍 Dataset Highlights

- ✅ Balanced splits (train/val/test)
- ✅ Source diversity across video origins
- ✅ Strong augmentations for rare classes (dogs, wolves, horses) using Albumentations
- ✅ Validated through entropy metrics, class histograms, and recycling diagnostics

### 📁 Structure

herding_dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
├── labels/
│ ├── train/
│ ├── val/
│ └── test/
├── metadata/ # CSV reports: class distribution, augmentation logs
├── visualizations/ # Entropy plots, class balance
├── dataset_config.yaml # Ready for Ultralytics training
└── README.md

The dataset is compatible with:
- 🧠 **YOLOv5–YOLOv12**
- 🔁 **RT-DETR**
- 🐑 **SHEPHERD**
- 🤖 **ROS 2 (Humble)** on quadrupeds (Unitree Go1 tested)

More info & generation pipeline: [dataset_strategy on GitHub](https://github.com/JeanMayoko18/yolo_shepherd/tree/main/dataset_strategy)

## 🚀 Getting Started

1. Clone the Repository
```bash
git clone https://github.com/JeanMayoko18/yolo_shepherd.git
cd yolo_shepherd
```bash
3. Prepare Dataset

Ensure the dataset is placed in datasets/herding_dataset/ with the structure above. You can also generate your own using the provided dataset_strategy tools.

📊 Performance Comparison

| Model        | mAP\@0.5 (Test) | mAP\@0.5 (Val) | FPS       | Size       |
| ------------ | --------------- | -------------- | --------- | ---------- |
| YOLOv8n      | 54.9%           | 69.4%          | 299.7     | 6.23MB     |
| YOLOv11      | 55.0%           | 69.7%          | 271.5     | 8.16MB     |
| **SHEPHERD** | **57.1%**       | **72.0%**      | **292.9** | **6.29MB** |
| RTDETR-L     | 53.2%           | 64.5%          | 94.1      | 123MB      |

## Citation

If you use this repository in your work, please cite:
@article{mayoko2025shepherd,
  title={SHEPHERD: Lightweight Real-Time Detection of Sheep and Predators for Autonomous Livestock Monitoring},
  author={Mayoko, J. C. and Sánchez González, L. and Kafunda, P. and Francisco J. R.},
  journal={},
  year={2025},
  pages ={1 - 30}
}

#Acknowledgments
This project has been partially funded by the Recovery, Transformation, and Resilience Plan, financed by the European Union (Next Generation), thanks to the AURORAS project, and by grant PID2024-161761OB-C21 funded by MICIU/AEI/10.13039/501100011033 and by the ERDF/EU.

<div style="text-align:center;">
  <img src="https://project-auroras.github.io/auroras/logos/tituloblanco.png" alt="Logo AURORAS" width="500" style="margin-right:20px;"/>
  <img src="logos/MICIUCofinanciadoAEI.jpg" alt="Logo Funding" width="500"/>
</div>
