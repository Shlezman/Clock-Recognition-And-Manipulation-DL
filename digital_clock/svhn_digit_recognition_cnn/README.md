# SVHN CNN Model — Complete Technical Description

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Model Architecture](#4-model-architecture)
5. [Key Building Blocks](#5-key-building-blocks)
6. [Activation Functions](#6-activation-functions)
7. [Regularization Techniques](#7-regularization-techniques)
8. [Weight Initialization](#8-weight-initialization)
9. [Loss Function](#9-loss-function)
10. [Optimizer](#10-optimizer)
11. [Learning Rate Scheduler](#11-learning-rate-scheduler)
12. [Accuracy Metric](#12-accuracy-metric)
13. [Training Pipeline](#13-training-pipeline)
14. [Hyperparameters Summary](#14-hyperparameters-summary)
15. [Training Results](#15-training-results)
16. [Model Evaluation](#16-model-evaluation)

---

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** in PyTorch to solve the **Street View House Numbers (SVHN)** recognition task. The goal is not simply to classify a single digit, but to recognize an entire **sequence of up to 5 digits** from a single 32×32 grayscale image. For example, a house number "347" must be identified as the sequence `[3, 4, 7, 10, 10]`, where `10` is a special padding token indicating "no digit present."

This is a multi-label sequence prediction problem — significantly more challenging than standard single-digit classification.

---

## 2. Dataset

The data is loaded from a preprocessed HDF5 file (`SVHN_unified_norm_gray.h5`). The dataset is split into three parts:

| Split      | Samples  | Purpose                                      |
|------------|----------|----------------------------------------------|
| Training   | 243,679  | Used to update the model's weights           |
| Validation | 27,076   | Used to monitor performance during training  |
| Test       | 14,374   | Used for final, unbiased performance report  |

Each image is a **32×32 grayscale** image (1 channel), and each label is a vector of 5 integers representing the digit sequence. The class `10` is reserved as a "null" or padding value for positions where no digit exists (e.g., a 2-digit number only fills positions 1 and 2; positions 3–5 are labeled `10`). This gives us **11 possible classes** per digit position (digits 0–9 plus the null class `10`).

---

## 3. Data Preprocessing

Before training, the images undergo two important transformations:

**Grayscale Conversion:** The images have already been converted to a single grayscale channel during the dataset preparation stage. This reduces memory and computation without significantly impacting accuracy for digit recognition, as shape and contrast carry more information than color.

**Normalization:** Pixel values are normalized so that they fall within a consistent numerical range. Neural networks train much more effectively and stably when input values are small and zero-centered, rather than raw pixel values in the range [0, 255].

**Channel Transposition:** PyTorch expects image tensors in the format `(Batch, Channels, Height, Width)`, denoted as NCHW. The HDF5 file stores images as `(Batch, Height, Width, Channels)` (NHWC, the TensorFlow/NumPy convention). The code transposes the axes using `np.transpose(X, (0, 3, 1, 2))` to convert between these formats before training.

---

## 4. Model Architecture

The model (`SVHNModel`) follows a classic **CNN + Multi-Head Fully Connected** design. It consists of three convolutional blocks for feature extraction, followed by shared fully connected layers, and finally five independent output heads — one per digit position.

```
Input: (Batch, 1, 32, 32) — grayscale image

┌──────────────────────────────────────────────┐
│  BLOCK 1: Feature Extraction (Low-Level)      │
│  Conv1: 1  → 32 filters, 5×5, no pooling     │
│  Conv2: 32 → 48 filters, 5×5, AvgPool (÷2)  │
│  Dropout (p=0.10)                             │
│  Output spatial size: 16×16                  │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│  BLOCK 2: Feature Extraction (Mid-Level)      │
│  Conv3: 48 → 64 filters, 5×5, no pooling    │
│  Conv4: 64 → 80 filters, 5×5, AvgPool (÷2)  │
│  Dropout (p=0.10)                             │
│  Output spatial size: 8×8                    │
└──────────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────┐
│  BLOCK 3: Feature Extraction (High-Level)     │
│  Conv5: 80  → 96  filters, 5×5, no pooling   │
│  Conv6: 96  → 112 filters, 5×5, no pooling   │
│  Conv7: 112 → 128 filters, 5×5, AvgPool (÷2) │
│  Dropout (p=0.50)                             │
│  Output spatial size: 4×4                    │
└──────────────────────────────────────────────┘
                      ↓
         Flatten → 4×4×128 = 2,048 neurons
                      ↓
┌──────────────────────────────────────────────┐
│  SHARED FULLY CONNECTED LAYERS               │
│  FC1: 2,048 → 1,024 neurons + LeakyReLU      │
│  Dropout (p=0.50)                             │
│  FC2: 1,024 → 1,024 neurons + LeakyReLU      │
└──────────────────────────────────────────────┘
                      ↓
       ┌──────┬──────┬──────┬──────┬──────┐
       ↓      ↓      ↓      ↓      ↓      
  Digit1  Digit2  Digit3  Digit4  Digit5
  1024→11 1024→11 1024→11 1024→11 1024→11
       
Output: 5 vectors of 11 logits each
```

**Why 5 separate output heads?** Because the model predicts a *sequence* rather than a single digit. Each head specializes in predicting the digit at a specific position in the number sequence. This "multi-head" design is more efficient than training five completely separate models, because all heads share and benefit from the same feature representations learned by the convolutional blocks and FC layers.

---

## 5. Key Building Blocks

### `ConvLayer`
This is a custom reusable block that wraps several operations together:

1. **Conv2d:** A 2D convolution with kernel size 5×5 and `SAME` padding (padding = `(kernel_size - 1) // 2 = 2`). This ensures the spatial dimensions are preserved after convolution when stride=1.
2. **BatchNorm2d:** Batch Normalization (explained below).
3. **LeakyReLU:** The non-linear activation function (explained below).
4. **AvgPool2d (optional):** A 2×2 average pooling layer that halves the spatial dimensions. Only some layers use pooling.

### `FCLayer`
A simple wrapper around a fully connected (linear) layer, optionally followed by a LeakyReLU activation. The output heads use `relu=False` because their raw outputs (logits) are passed directly to the Cross Entropy loss.

### Batch Normalization
Batch Normalization (BatchNorm) is applied after every convolution. During training, it normalizes the output of the convolution across the current mini-batch so that the values have a mean of 0 and a standard deviation of 1. It then applies learnable scale (`gamma`) and shift (`beta`) parameters.

**Why use it?** It stabilizes training by preventing the internal distribution of activations from shifting dramatically as the model updates (a problem called "internal covariate shift"). It acts as a mild regularizer and often allows for higher learning rates, leading to faster convergence.

### Average Pooling
After every two (or three) convolutional layers, an `AvgPool2d(2, 2)` layer is applied. It slides a 2×2 window over the feature map and replaces each window with the average of the four values. This halves the spatial dimensions (e.g., 32×32 → 16×16).

**Why pool?** It reduces the number of parameters and computations in subsequent layers. It also introduces a small amount of translational invariance — the model becomes less sensitive to the exact pixel location of a feature.

**Why Average Pooling instead of Max Pooling?** Average pooling is a gentler operation that retains information about the entire receptive field, not just the dominant feature. It is often preferred in deeper networks where the information density is high.

---

## 6. Activation Functions

### Leaky ReLU (Negative Slope = 0.10)
An **activation function** introduces non-linearity into the network. Without it, stacking many linear layers would be mathematically equivalent to a single linear transformation — the network would be unable to learn complex patterns.

The standard **ReLU** (Rectified Linear Unit) is defined as `f(x) = max(0, x)`. While highly effective, it has a flaw: any neuron receiving a consistently negative input will always output zero and its gradient will be zero, meaning it can never recover and contribute to learning. This is known as the "dying ReLU" problem.

**Leaky ReLU** fixes this by allowing a small, non-zero gradient for negative inputs:

```
f(x) = x          if x > 0
f(x) = 0.10 * x   if x ≤ 0
```

With a `negative_slope` of `0.10`, negative inputs still produce a small output instead of being completely zeroed out. This keeps all neurons "alive" and contributing to learning throughout training.

---

## 7. Regularization Techniques

Regularization refers to any technique that prevents a model from **overfitting** — memorizing the training data instead of learning generalizable patterns.

### Dropout
Dropout is applied at several points in the network. During each training step, it randomly sets a fraction of neuron outputs to zero. The fraction zeroed out is controlled by the probability `p`:

| Location                 | Dropout Rate (p) | Notes                                      |
|--------------------------|------------------|--------------------------------------------|
| After Block 1            | 0.10 (10%)       | Light regularization on early features     |
| After Block 2            | 0.10 (10%)       | Light regularization on mid-level features |
| After Block 3            | 0.50 (50%)       | Heavy regularization on deep features      |
| Between FC1 and FC2      | 0.50 (50%)       | Heavy regularization in the dense layers   |

The intuition behind Dropout is that it forces the network to learn redundant representations of features. No single neuron can be "relied upon," so the network builds more robust and distributed representations. During evaluation (inference), Dropout is turned off (`model.eval()`) and all neurons are active.

The higher dropout rate (50%) in the deeper layers is intentional — at that stage, the network has far more parameters and is at greater risk of overfitting.

---

## 8. Weight Initialization

Before training begins, the model's weights must be set to some initial values. Poor initialization can cause training to be very slow or fail entirely (e.g., due to vanishing or exploding gradients).

This model uses **He (Kaiming) Initialization** for convolutional layers (with `method='he'`) and optionally **Xavier Initialization** for others.

### He (Kaiming) Initialization
Designed for layers followed by ReLU-like activations. It initializes weights from a uniform distribution scaled by `sqrt(2 / fan_in)`, where `fan_in` is the number of input connections. This ensures that the variance of the activations remains consistent across layers.

### Xavier (Glorot) Initialization
A more general initialization method. It scales weights by `sqrt(2 / (fan_in + fan_out))` and is well-suited for symmetric activations like `tanh` or `sigmoid`. Used here for fully connected layers.

In both cases, **biases are initialized to 0**, which is standard practice.

---

## 9. Loss Function

The model uses **Cross-Entropy Loss** (`nn.CrossEntropyLoss`), the most common loss function for multi-class classification problems.

For a single prediction, the loss measures how different the model's output probability distribution is from the true label. Internally, PyTorch's `CrossEntropyLoss` combines a `Softmax` function (which converts raw logits into probabilities) and the Negative Log-Likelihood Loss into a single, numerically stable operation.

**Softmax** converts a vector of raw scores (logits) `z` into probabilities:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
```

The **Negative Log-Likelihood** then penalizes the model more heavily when it assigns a low probability to the correct class:

```
NLL Loss = -log(P(correct class))
```

If the model is very confident and correct, `P(correct class) ≈ 1.0` and the loss ≈ 0. If the model is confident but wrong, `P(correct class) ≈ 0.0` and the loss → ∞.

### Multi-Head Loss Aggregation
Since the model has 5 output heads, the total loss for a training step is the **sum** of the individual Cross-Entropy losses for each of the 5 digit positions:

```
Total Loss = CE(ŷ₁, y₁) + CE(ŷ₂, y₂) + CE(ŷ₃, y₃) + CE(ŷ₄, y₄) + CE(ŷ₅, y₅)
```

This means the model is simultaneously trained to be correct on all 5 positions at once. By summing (rather than averaging), each head's error contributes equally to the gradient signal, and the total loss scale is proportionally larger when more digits are wrong.

---

## 10. Optimizer

The model uses the **Adam optimizer** (`optim.Adam`) with a learning rate of `1e-3` (0.001).

An optimizer is the algorithm that updates the model's weights based on the computed gradients. Simple **Stochastic Gradient Descent (SGD)** updates weights in proportion to the gradient: `w = w - lr * gradient`. This works but can be slow and sensitive to the choice of learning rate.

**Adam (Adaptive Moment Estimation)** is a much more sophisticated optimizer that maintains two running averages for each parameter:

- **m (first moment):** An exponential moving average of the gradients (like momentum). This helps accelerate learning in consistent directions.
- **v (second moment):** An exponential moving average of the *squared* gradients. This tracks how large the gradients typically are for each parameter.

The weight update rule is:
```
w = w - (lr / sqrt(v) + ε) * m
```

In effect, Adam adapts the learning rate for *each individual parameter* based on its historical gradient magnitudes. Parameters with consistently large gradients get a smaller effective learning rate; parameters with small gradients get a larger one. This makes Adam robust, fast-converging, and a very popular default choice for deep learning.

---

## 11. Learning Rate Scheduler

The optimizer uses a **StepLR scheduler** (`optim.lr_scheduler.StepLR`) to gradually reduce the learning rate during training.

```
DECAY_STEPS = 8800   # Reduce LR every 8,800 optimizer steps
DECAY_GAMMA = 0.5    # Multiply the LR by 0.5 at each step
```

**Why reduce the learning rate?** Early in training, a large learning rate allows the model to make rapid progress. As training progresses and the model approaches a good solution, a large learning rate can cause the weights to "bounce around" and overshoot the optimal values. Reducing the LR over time allows the model to make finer, more precise adjustments in the later stages of training, helping it converge to a better final solution.

The scheduler is called at *every training step* (not every epoch), making the decay very gradual and smooth relative to the total number of steps.

---

## 12. Accuracy Metric

The accuracy is computed using a **strict sequence-level metric**: a prediction is only considered correct if **all 5 digit positions are correctly predicted simultaneously**.

```python
# Stack all 5 output logits: shape [Batch, 5, 11]
# Find the argmax (predicted class) for each position
predictions = torch.argmax(stacked_logits, dim=2)  # shape [Batch, 5]

# A sample is "correct" only if ALL 5 positions match the label
correct_sequences = (predictions == labels).all(dim=1)
accuracy = correct_sequences.float().mean() * 100.0
```

This is a deliberately strict metric. For a 3-digit number like "347," predicting "348" scores 0% — not 66%. This makes the reported accuracy harder to achieve than per-digit accuracy, but it is a more faithful measure of whether the model can actually read house numbers correctly.

---

## 13. Training Pipeline

The `train_process()` function orchestrates the full training lifecycle:

**Step 1 — Data Loading:** The HDF5 file is read, transposed to NCHW format, and wrapped in PyTorch `TensorDataset` objects. These are then passed to `DataLoader`, which handles shuffling (training set only) and batching into mini-batches of 512 images.

**Step 2 — Initialization:** The model, Adam optimizer, StepLR scheduler, and CrossEntropy loss criterion are instantiated.

**Step 3 — Checkpoint Restoration:** The code checks for previously saved `.pth` files. If one exists, the model and optimizer states are loaded and training resumes from where it left off.

**Step 4 — The Training Loop:** For each of the 20 epochs:
- The model is set to **train mode** (`model.train()`), which enables Dropout and BatchNorm's training-time behavior.
- For each mini-batch:
  1. **Zero Gradients:** `optimizer.zero_grad()` clears gradients from the previous step to prevent accumulation.
  2. **Forward Pass:** The batch of images is passed through the network to produce 5 sets of logits.
  3. **Loss Calculation:** The summed Cross-Entropy loss is computed.
  4. **Backward Pass:** `loss.backward()` computes gradients via **backpropagation** — the chain rule of calculus applied to the entire computation graph.
  5. **Weight Update:** `optimizer.step()` applies the Adam update rule to all parameters.
  6. **Scheduler Step:** The learning rate is potentially decayed.

**Step 5 — Validation:** At the end of each epoch, the model is switched to **eval mode** (`model.eval()`) and evaluated on the held-out validation set. Gradients are not computed (`torch.no_grad()`), saving memory.

**Step 6 — Checkpointing:** The model weights and optimizer state are saved to a `.pth` file after each epoch.

**Step 7 — Final Test:** After all epochs, the model is evaluated on the test set for a final, unbiased performance estimate.

**TensorBoard Logging:** Training loss and accuracy are logged at every 100 steps, and validation accuracy is logged at every epoch, enabling real-time visualization of the training curves.

---

## 14. Hyperparameters Summary

| Hyperparameter         | Value                          | Description                                  |
|------------------------|--------------------------------|----------------------------------------------|
| `BATCH_SIZE`           | 512                            | Samples per mini-batch                       |
| `EPOCHS`               | 20                             | Full passes over the training data           |
| `LEARNING_RATE`        | 1e-3 (0.001)                   | Initial Adam learning rate                   |
| `DECAY_STEPS`          | 8,800                          | LR halved every this many optimizer steps    |
| `DECAY_GAMMA`          | 0.5                            | LR multiplier at each decay event            |
| `FILTER_SIZES`         | [5, 5, 5, 5, 5, 5, 5]         | Kernel size for each conv layer              |
| `NUM_FILTERS`          | [32, 48, 64, 80, 96, 112, 128] | Number of filters per conv layer             |
| `FC1_SIZE`             | 1,024                          | Neurons in first fully connected layer       |
| `FC2_SIZE`             | 1,024                          | Neurons in second fully connected layer      |
| `NUM_LABELS`           | 11                             | Classes per digit (0–9 + null class 10)      |
| `LeakyReLU slope`      | 0.10                           | Negative slope for Leaky ReLU                |
| Dropout (conv blocks)  | 0.10                           | After blocks 1 and 2                         |
| Dropout (deep layers)  | 0.50                           | After block 3 and between FC1/FC2            |
| Weight Init (conv)     | He (Kaiming)                   | Suited for Leaky ReLU activations            |
| Weight Init (FC)       | Xavier (Glorot)                | General purpose initialization               |

---

## 15. Training Results

The model was trained on a CUDA GPU and completed in approximately **6 minutes** (about 17–18 seconds per epoch).

| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 67.25%              |
| 2     | 84.67%              |
| 3     | 87.27%              |
| 5     | 90.40%              |
| 9     | 92.21%              |
| 12    | 93.06%              |
| 17    | 93.30%              |
| 19    | 94.04%              |
| **20**| **94.13%**          |

**Final Test Accuracy: 92.97%**

The training curve shows very rapid initial convergence (from ~0% to ~67% in just the first epoch), followed by steady, consistent improvement. The gap between validation (94.13%) and test (92.97%) accuracy is small, suggesting the model generalizes well without significant overfitting. This is in large part thanks to the Dropout layers and Batch Normalization.

---

## 16. Model Evaluation

Beyond the quantitative accuracy, the notebook includes a rich qualitative evaluation:

**Confusion Matrices:** Plotted per digit position, showing which digits the model most often confuses with each other (e.g., confusing `1` with `7` or `6` with `8`).

**Sample Visualizations:** A grid of test images is displayed alongside their true labels and the model's predictions, with correct predictions highlighted in green and incorrect ones in red. This allows for intuitive error analysis — for instance, identifying that the model struggles with low-resolution, noisy, or partially occluded images.

**Label-to-String Conversion:** The evaluation code includes a utility that converts raw label tensors (e.g., `[3, 4, 7, 10, 10]`) into human-readable strings (`"347"`), stripping the null-padding tokens for clean display.