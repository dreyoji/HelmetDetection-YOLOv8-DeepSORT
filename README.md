# Helmet Detection using YOLOv8, DeepSORT, and EfficientNET

---

## Overview

This repository contains code for detecting helmets in images using the YOLOv8 object detection algorithm. Two datasets, one for good weather conditions and one for bad weather conditions, were utilized for training and testing the model. The datasets were annotated using Roboflow and exported in Jupyter Notebook format. Image enhancement techniques, specifically Contrast Limited Adaptive Histogram Equalization (CLAHE), were applied to improve the quality of the images before training.

## Comparison

Two different approaches to helmet detection were compared:

- **YOLOv8 + DeepSORT**: This method incorporates object tracking using the DeepSORT algorithm, enabling the tracking of helmets across frames in a video or sequence of images.

- **YOLOv8 + EfficientNET**: Here, the YOLOv8 model is augmented with the EfficientNET architecture, leveraging its capabilities for improved feature extraction and detection 