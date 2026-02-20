# Fish Detector 🐟

**Fish Detector** is an object detection project that identifies and classifies marine creatures such as fish, jellyfish, penguins, sharks, puffins, starfish, and stingrays using the **YOLOv8** deep learning model. The project uses the **Aquarium Dataset (COTS)** from Kaggle.

---

## Dataset

The project uses the [Aquarium Data COTS dataset](https://www.kaggle.com/slavkoprytula/aquarium-data-cots) from Kaggle, which contains labeled images of various marine animals.

* The dataset includes training and validation splits.
* It covers multiple species commonly found in aquariums.

---

## Project Overview

* The model is based on **YOLOv8** for real-time object detection.
* Pretrained weights are used to speed up training and improve accuracy.
* Data augmentation techniques (e.g., mosaic, mixup) are applied to enhance model generalization.
* The model is trained for **150 epochs** with multi-scale images.

---

## Features

* Detects **7 different classes** of marine creatures:

  * Fish
  * Jellyfish
  * Penguin
  * Puffin
  * Shark
  * Starfish
  * Stingray

* Provides real-time detection capability.

* Supports visualization of bounding boxes and confidence scores.

---

## Model Performance

After training, the model achieves the following performance on the validation set:

| Class     | Precision | Recall | mAP50 | mAP50-95 |
| --------- | --------- | ------ | ----- | -------- |
| Fish      | 0.81      | 0.747  | 0.802 | 0.466    |
| Jellyfish | 0.84      | 0.879  | 0.919 | 0.524    |
| Penguin   | 0.644     | 0.645  | 0.666 | 0.316    |
| Puffin    | 0.572     | 0.434  | 0.506 | 0.264    |
| Shark     | 0.758     | 0.614  | 0.671 | 0.42     |
| Starfish  | 0.777     | 0.667  | 0.733 | 0.527    |
| Stingray  | 0.661     | 0.758  | 0.723 | 0.505    |

* The model is optimized for **both speed and accuracy**, making it suitable for real-time applications.

---

## Results Visualization

* Training outputs include labeled images, bounding boxes, and predictions.
* Users can visualize results for both training and inference.

---

## Requirements

* Python 3.8+
* PyTorch
* Ultralytics YOLOv8
* KaggleHub (for dataset download)

---

## Usage

* Train the model on the Aquarium Dataset.
* Detect and classify marine creatures in new images or videos.
* Visualize bounding boxes and confidence scores for predictions.

---

## Notes

* The project leverages **pretrained YOLOv8 weights** for faster convergence.
* Data augmentation is applied to improve model generalization.
* Hyperparameters (image size, batch size, epochs) can be adjusted based on GPU availability and dataset size.
