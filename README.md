# Chest X-ray Pneumonia Classification with Explainable AI

An end-to-end deep learning pipeline for binary chest X-ray classification (Normal vs. Pneumonia) using **PyTorch**. This project fine-tunes a pretrained **ResNet18** model and demonstrates multiple **Explainable AI (XAI)** techniques to visualize the image regions that contribute to the model's predictions.

---

## Overview

Deep learning models have achieved remarkable performance in medical image analysis, but their lack of interpretability remains a major challenge. This project combines accurate chest X-ray classification with several Explainable AI methods to help understand how the model makes its decisions.

The notebook covers the complete workflow from data preprocessing and model training to evaluation and explainability visualization.

📓 **Kaggle Notebook:** [View Notebook](https://www.kaggle.com/code/atenazare/chest-x-ray-classification)
---

## Features

- Fine-tuning a pretrained ResNet18 model
- Data preprocessing and augmentation
- Training and validation pipeline
- Binary classification of chest X-ray images
- Performance evaluation with multiple metrics
- Visualization of predictions
- Explainable AI using:
  - Grad-CAM
  - Integrated Gradients
  - Occlusion
  - Layer-wise Relevance Propagation (LRP)

---

## Dataset

This project uses the **Chest X-ray Pneumonia** dataset.

Directory structure:

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Dataset:** [View dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## Model

- Architecture: ResNet18
- Framework: PyTorch
- Transfer Learning: ImageNet pretrained weights
- Task: Binary Image Classification

**Trained model:** [View model](https://www.kaggle.com/models/atenazare/finetuned-resnet18-chestxray/PyTorch/default/1)
---

## Explainable AI Methods

This repository demonstrates several attribution techniques for interpreting model predictions.

### Grad-CAM

Highlights the most important spatial regions that influence the prediction.

### Integrated Gradients

Computes feature attributions by integrating gradients from a baseline image to the input image.

### Occlusion

Measures feature importance by systematically masking different image regions and observing changes in prediction confidence.

---

## Evaluation

The notebook includes:

- Accuracy
- Precision
- Confusion Matrix
- Classification Report

---

## Technologies

- Python
- PyTorch
- Torchvision
- Captum
- NumPy
- Matplotlib
- scikit-learn
- OpenCV
- PIL

---

## Repository Structure

```
.
├── Chest_Xray_Classification.ipynb
├── README.md
├── results/
```

---

## Example Outputs

The notebook generates visual explanations including:

- Original Chest X-ray
- Prediction
- Grad-CAM heatmap
- Integrated Gradients attribution
- Occlusion attribution
- Overlay visualizations

---

## Future Improvements

- Vision Transformer (ViT) models
- EfficientNet and ConvNeXt comparison
- Multi-class thoracic disease classification
- Quantitative comparison of XAI methods
- Clinical validation of explanations

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision captum grad-cam scikit-learn matplotlib numpy pillow opencv-python
```

---

## Citation

If you find this repository useful, please consider giving it a ⭐ on GitHub.

---

## Author

**Fatemeh (Atena) Zare**

Software Engineer | Deep Learning | Computer Vision | Explainable AI
