# 🧠 Technical Documentation: AI/ML Feature Implementation

This document provides a comprehensive deep dive into the Machine Learning logic, model architecture, and training pipeline used in the **AI vs Real Image Classifier**.

---

## 1. Problem Statement
The goal is to perform **Binary Classification** on images to determine their origin:
- **Class 0**: AI Generated
- **Class 1**: Real Image

Given the visual complexity and high quality of modern AI generation (Midjourney, Stable Diffusion), simple CNNs often fail to capture subtle artifacts. Therefore, we utilize **Transfer Learning** with a state-of-the-art architecture.

---

## 2. Model Architecture
We employ **EfficientNetB0** as our backbone. EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.

### 2.1 Why EfficientNetB0?
- **Efficiency**: It achieves higher accuracy with fewer parameters compared to ResNet50 or VGG16.
- **Feature Extraction**: Pre-trained on **ImageNet** (1000 classes), it has already learned robust low-level features (edges, textures) and high-level features (objects, patterns) which are transferable to our task.
- **Input Size**: Optimized for $224 \times 224$ pixel inputs.

### 2.2 Custom Classification Head
The pre-trained top layer of EfficientNet is removed, and replaced with a custom head designed for our binary task:

```mermaid
graph TD
    Input[Input Image (224x224x3)] --> Aug[Data Augmentation Layer]
    Aug --> Base[EfficientNetB0 (Frozen/Unfrozen)]
    Base --> GAP[GlobalAveragePooling2D]
    GAP --> BN[BatchNormalization]
    BN --> Drop[Dropout (0.3)]
    Drop --> Dense[Dense Output (1 Unit, Sigmoid)]
```

- **GlobalAveragePooling2D**: Reduces the spatial dimensions ($7 \times 7 \times 1280$) to a vector ($1280$), minimizing overfitting compared to Flattening.
- **BatchNormalization**: Stabilizes learning and accelerates convergence.
- **Dropout (0.3)**: Randomly sets 30% of inputs to 0 during training to prevent overfitting.
- **Sigmoid Activation**: Outputs a probability score between 0 and 1.
  - $P(x) < 0.5 \Rightarrow \text{AI Generated}$
  - $P(x) \ge 0.5 \Rightarrow \text{Real Image}$

---

## 3. Training Strategy (Transfer Learning)
We implement a **Two-Phase Training Strategy** to maximize performance while maintaining the stability of pre-trained weights.

### Phase 1: Feature Extraction
- **State**: `EfficientNetB0` base is **FROZEN** (non-trainable).
- **Goal**: Train only the new custom head (Dense layer) to adapt to the features output by the base model.
- **Hyperparameters**:
  - `Learning Rate`: $1 \times 10^{-3}$ (Standard Adam default)
  - `Epochs`: 10
- **Logic**: If we train the whole model immediately with random weights in the head, the large error gradients would destroy the delicate pre-trained weights of the base model.

### Phase 2: Fine-Tuning
- **State**: `EfficientNetB0` base is **UNFROZEN** (trainable).
- **Goal**: Adjust the deeper layers of the efficientnet to learn specific artifacts of AI generation (e.g., subtle pixel inconsistencies).
- **Hyperparameters**:
  - `Learning Rate`: $1 \times 10^{-5}$ (Very Low!)
  - `Epochs`: +10 (Total 20)
- **Logic**: A low learning rate is crucial here to gently nudge the weights without un-learning the robust ImageNet features.

---

## 4. Data Pipeline
The robust data pipeline handles loading, preprocessing, and augmentation.

### 4.1 Data Augmentation
To prevent the model from memorizing specific images, we apply random transformations during training:
- **RandomFlip**: Horizontal mirroring.
- **RandomRotation**: $\pm 20\%$.
- **RandomZoom**: $\pm 20\%$.
- **RandomContrast**: $\pm 20\%$.

This ensures the model learns **invariant features** rather than specific pixel arrangements.

### 4.2 Class Imbalance Handling
Real-world datasets are often imbalanced. We compute **Class Weights** dynamically:
$$ w_j = \frac{\text{n\_samples}}{2 \times \text{n\_samples}_j} $$
These weights are passed to the loss function, penalizing the model more for misclassifying the minority class, ensuring unbiased learning.

---

## 5. Evaluation Metrics
We rely on more than just accuracy to validate the model:

1. **Confusion Matrix**:
   - **True Positives (TP)**: Real images correctly identified as Real.
   - **False Negatives (FN)**: Real images wrongly flagged as AI (Type II Error).
   - **False Positives (FP)**: AI images wrongly flagged as Real (Type I Error).
   - **True Negatives (TN)**: AI images correctly identified as AI.

2. **ROC Curve & AUC**:
   - Plots **True Positive Rate (TPR)** vs **False Positive Rate (FPR)** at various threshold settings.
   - An **AUC** (Area Under Curve) close to 1.0 indicates a perfect classifier.

---

## 6. Implementation Specifications
- **Framework**: TensorFlow / Keras 2.10+
- **Input Resolution**: $224 \times 224 \times 3$ (RGB)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Hardware Acceleration**: Metal (MPS) on Mac / CUDA on NVIDIA.

---
*Documentation generated for the Classifier Project.*
