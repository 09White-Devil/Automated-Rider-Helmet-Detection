# Automated-Rider-Helmet-Detection
# 🏍️ YOLOv8m Hierarchical Object Detection for Road Safety Enforcement

## Project Overview
This repository contains the full source code, trained model weights, and detailed documentation for an **Automated Helmet and Rider Detection System**. The system is designed to monitor traffic streams in real-time to detect motorcycle riders and assess helmet compliance.

We engineered a custom **Hierarchical 3-Class Labeling Strategy** and trained a **YOLOv8m** model on a massive, combined dataset of over 21,000 images, prioritizing **robustness and generalization** over purely theoretical maximum accuracy.

---

## 🚀 Key Achievements & Results

| Metric | Value | Technical Context |
| :--- | :--- | :--- |
| **Final mAP@50** | **82.2%** | **Excellent Reliability** on a highly diverse, uncurated dataset. |
| **Precision** | **85.9%** | **High Trust Score.** The model minimizes false positives (false alarms), crucial for enforcement applications. |
| **Recall** | **75.2%** | **Good Coverage.** Indicates the model finds 75% of all objects in complex scenes. |
| **Total Dataset Size** | **21,862 Images** | Achieved **State-of-the-Art scale** by aggregating 5 distinct public sources. |
| **Model** | **YOLOv8m** | 25.9 Million trainable parameters. |
| **Deployment Focus**| **TensorRT Ready** | Defined pipeline for conversion to TensorRT, enabling real-time inference (>30 FPS) on NVIDIA edge devices. |

---

## 🛠️ Methodology & Technical Stack

### 1. Data Engineering (The Ultimate Dataset)
The project's strength lies in its **data strategy**. We moved from a simple 2-class problem to a contextual 3-class hierarchy.

* **Final Classes:** `rider`, `helmet`, `no helmet`.
* **Strategy:** We merged five distinct public datasets (Roboflow Universe, Kaggle) to maximize diversity in lighting, angles, and rider posture.
* **Hierarchical Annotation:** The model was explicitly taught the relationship between the objects: the `helmet` or `no helmet` bounding box is nested within the `rider` bounding box, which greatly reduces confusion with background objects.

### 2. Architecture & Training
* **Framework:** PyTorch (via Ultralytics YOLOv8 Library).
* **Model:** **YOLOv8m** (Medium) initialized via **Transfer Learning** from the COCO dataset.
* **Key Modules:** Utilized **C2f modules** in the Backbone and the **PANet Neck** for multi-scale feature fusion.
* **Training Configuration:**
    * **Epochs:** 150 (with Early Stopping).
    * **Image Size:** 512x512.
    * **Augmentations:** Aggressive augmentation applied directly in the training pipeline: **HSV shifts** (0.7 Saturation, 0.4 Value) and **Horizontal Flipping** (0.5 probability) to simulate real-world environmental changes.
* **Loss Functions:** Minimized composite loss combining **CIoU** (Complete Intersection over Union) for box precision and **DFL** (Distribution Focal Loss) for accurate boundary prediction.

### 3. Reproducibility & Deployment
The model was trained using the following final command and is ready for optimized deployment:

```bash
# Final Training Command (Generating best.pt)
yolo detect train data="data/Ultimate_Data/data.yaml" model=yolov8m.pt epochs=150 imgsz=512 name=final_sota_150epochs batch=16 augment=True hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 patience=25

# Deployment Path
yolo export model=runs/detect/final_sota_150epochs/weights/best.pt format=engine 
# This converts the model to NVIDIA TensorRT format for maximum speed on the RTX 4050 GPU.
