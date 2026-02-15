# Stress Testing of CNN (ResNet-18 Implementation in PyTorch)

## Overview

This project implements a Custom ResNet-18 Convolutional Neural Network from scratch in PyTorch and evaluates its performance on the FashionMNIST dataset.
The goal is to study CNN behaviour under training stress, monitor learning curves, and evaluate classification accuracy.

The project includes:

* Custom Residual Block implementation
* Full ResNet-18 architecture built manually
* Additional experimental 3-Residual-Block model
* Training / Validation / Test pipeline
* Accuracy and loss tracking across epochs
* GPU auto-detection
  
## Model Architecture
Custom ResNet-18 Structure
Input (1×28×28)
   ↓
Conv(1 → 64)
   ↓
Residual Layer1 → 64 filters (2 blocks)
Residual Layer2 → 128 filters (2 blocks)
Residual Layer3 → 256 filters (2 blocks)
Residual Layer4 → 512 filters (2 blocks)
   ↓
Global Average Pooling
   ↓
Fully Connected Layer
   ↓
10 Classes Output
Residual Block Formula
Output = F(x) + Shortcut(x)

## Technologies Used

* Python 3
* PyTorch
* Torchvision
* NumPy
* Matplotlib

### Dataset
FashionMNIST
The model is trained on the FashionMNIST dataset, which contains grayscale clothing images.
| Property | Value |
|---|---|
| Image Size | 28 × 28 |
| Channels | 1 |
| Classes | 10 |
| Training Samples | 60,000 |
| Test Samples | 10,000 |

### Dataset Split Used

| Split | Samples |
|---|---|
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |



Install dependencies:

```
pip install torch torchvision matplotlib numpy
```


## How to Run

Start Jupyter Notebook:

```
jupyter notebook
```

Open:

```
Stress_Testing_of_CNN.ipynb
```

Run all cells sequentially.



## GPU Support

The notebook automatically checks GPU availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

If CUDA is available, training runs on GPU automatically.



## Residual Block Code

```python
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.ReLU()(out)
```



## Training Loop model_1

| Parameter        | Value              |
|------------------|--------------------|
| Epochs           | 15                 |
| Batch Size       | 128                |
| Optimizer        | SGD                |
| Learning Rate    | 0.1                |
| Momentum         | 0.9                |
| Loss Function    | CrossEntropyLoss   |

## Evaluation Model_1

 Test Accuracy: 88.17%

 ### Common Failure Patterns

| Failure Type | Description |
|-------------|------------|
| Similar Clothing Types | Model confuses visually similar classes like Shirt vs T-Shirt |
| Low Contrast Images | Dark or faint images reduce feature clarity |
| Folded / Occluded Items | Clothing shape distortion causes misclassification |
| Unusual Texture Patterns | Rare textures not well represented in training set |
## Grad-CAM Visualization (Model Explainability)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize  
which regions of the input image influenced the model’s prediction.

This helps interpret CNN decisions and verify whether the network focuses on meaningful clothing regions.



### Grad-CAM Purpose

| Benefit | Description |
|--------|-------------|
| Model Interpretability | Shows where CNN is looking |
| Debugging Tool | Detects wrong attention regions |
| Failure Analysis | Helps explain incorrect predictions |
| Research Validation | Demonstrates explainable AI usage |



 ## Training Loop Improved model
 | Parameter | Value |
|----------|------|
| Epochs | 15 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Scheduler Type | StepLR |
| Step Size | 5 epochs |
| Gamma | 0.1 |
| Purpose | Reduce learning rate periodically for stable convergence |
| Loss Function | CrossEntropyLoss |



 ## Evaluation Improved_model

 Test Accuracy: 93.70%
##  Final Performance Comparison

| Metric | Baseline | Improved |
|-------|----------|----------|
| Training Accuracy | 94.24 % | 99.97 % |
| Validation Accuracy | 92.08 % | 93.88 % |
| Test Accuracy | 88.17 % | 93.70 % |
| Convergence Speed | Medium | Faster |
| Overfitting | Slight | Reduced |


## Learning Outcomes

After completing this project, you will understand:

* How ResNet works internally
* Why skip connections improve deep networks
* How to build CNN architectures from scratch
* How to train models efficiently using GPU



## Author

* Kashish Sharma- M25EEI003
* Minna Maria- M25EEI004

