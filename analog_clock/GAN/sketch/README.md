# Sketch-Conditioned cGAN for Clock Image Synthesis — Full Model & Training Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts](#2-core-concepts)
   - [What Is a Conditional GAN (cGAN)?](#21-what-is-a-conditional-gan-cgan)
   - [What Is Sketch-to-Image Synthesis?](#22-what-is-sketch-to-image-synthesis)
   - [How Does This Differ from the Inpainting GAN?](#23-how-does-this-differ-from-the-inpainting-gan)
3. [Dataset](#3-dataset)
   - [Directory Structure](#31-directory-structure)
   - [ClockDataset Class](#32-clockdataset-class)
   - [Data Preprocessing & Transforms](#33-data-preprocessing--transforms)
   - [DataLoaders](#34-dataloaders)
4. [Model Architecture](#4-model-architecture)
   - [Generator — Deep U-Net](#41-generator--deep-u-net)
   - [UNetDown and UNetUp Modules](#42-unetdown-and-unetup-modules)
   - [The Role of Dropout in the Generator](#43-the-role-of-dropout-in-the-generator)
   - [Discriminator — Deep PatchGAN](#44-discriminator--deep-patchgan)
5. [Loss Functions](#5-loss-functions)
   - [GAN Loss (Adversarial Loss)](#51-gan-loss-adversarial-loss)
   - [Pixel-wise Loss (L1)](#52-pixel-wise-loss-l1)
   - [Total Generator Loss](#53-total-generator-loss)
   - [Discriminator Loss](#54-discriminator-loss)
6. [Weight Initialization](#6-weight-initialization)
7. [Optimizers](#7-optimizers)
8. [Training Loop](#8-training-loop)
   - [Step-by-Step Walkthrough](#81-step-by-step-walkthrough)
   - [Generator Training Step](#82-generator-training-step)
   - [Discriminator Training Step](#83-discriminator-training-step)
   - [Asymmetric Update Schedule](#84-asymmetric-update-schedule)
9. [Hyperparameters Summary](#9-hyperparameters-summary)
10. [Visualization & Checkpointing](#10-visualization--checkpointing)
11. [Comparison to the Inpainting GAN](#11-comparison-to-the-inpainting-gan)
12. [Key Design Decisions Explained](#12-key-design-decisions-explained)

---

## 1. Project Overview

This project trains a **Conditional Generative Adversarial Network (cGAN)** to synthesize realistic clock images from two simultaneous inputs:

1. **A source image** — an augmented or modified version of a clock photo (e.g., different lighting, style, or augmentation).
2. **A sketch image** — a line drawing or edge map of the clock, describing its structural layout (hand positions, numeral placement, tick marks).

**The goal:** Given these two inputs, generate a **photorealistic target clock image** that matches the sketch's structure while adopting the visual style and details expected of a real clock photograph.

**In plain terms:**  
The model learns to "color in" and "texture up" a sketch of a clock — understanding where the hands, numerals, and dial are from the sketch, and how a real clock should look from the source reference.

This is an **image-to-image translation** task: the model translates a pair of abstract inputs (source + sketch) into a full-detail photorealistic output. It follows the general **pix2pix** framework but with the added complexity of two conditioning inputs instead of one.

---

## 2. Core Concepts

### 2.1 What Is a Conditional GAN (cGAN)?

A standard GAN generates images from random noise with no control over the output. A **Conditional GAN (cGAN)** extends this by also feeding the generator (and discriminator) a **conditioning signal** — additional information that guides what the output should look like.

In a cGAN:
- The Generator takes both a noise vector (or an input image) **and** a condition as input, and produces output conditioned on that input.
- The Discriminator receives both the condition and the candidate image (real or fake) and must judge whether the candidate is a plausible real output *for that specific condition*.

This conditioning is what makes the output controllable. In this project, the condition is the pair of (source image, sketch) — the model doesn't just generate *any* clock, it generates a clock that is consistent with the sketch structure and the source appearance.

### 2.2 What Is Sketch-to-Image Synthesis?

Sketch-to-image synthesis is a specific type of image-to-image translation where a sparse, abstract line drawing is "rendered" into a photorealistic image. The sketch captures structural information (shapes, edges, outlines) but lacks color, texture, shading, and fine detail.

The network must learn:
- **Shape faithfulness:** The output image must follow the spatial layout described by the sketch.
- **Texture hallucination:** The model must fill in realistic textures and colors not present in the sketch.
- **Semantic understanding:** The model must recognize what the sketch elements represent (clock hands, numbers, etc.) and render them appropriately.

This requires a model that can simultaneously understand high-level structure and low-level visual detail — which is exactly what a deep U-Net with skip connections is designed for.

### 2.3 How Does This Differ from the Inpainting GAN?

While both models use a U-Net generator and PatchGAN discriminator in a pix2pix-style framework, they solve fundamentally different problems:

| Aspect | Inpainting GAN | Sketch cGAN |
|---|---|---|
| **Task** | Remove clock hands, reconstruct background | Synthesize full clock from sketch |
| **Inputs** | Source (3ch) + binary mask (1ch) = 4 channels | Source (3ch) + sketch (3ch) = 6 channels |
| **Conditioning** | Spatial mask indicating what to fix | Structural sketch guiding generation |
| **Output** | Same clock, hands erased | Fully rendered clock from sketch |
| **U-Net depth** | 4 encoder layers | 8 encoder layers |
| **Discriminator inputs** | Source+mask (4ch) + target (3ch) = 7ch | Source (3ch) + sketch (3ch) + target (3ch) = 9ch |
| **Pixel loss weight** | LAMBDA_L1 = 500 (with mask weighting) | L1_LAMBDA = 1000 (uniform) |
| **Discriminator updates** | Every batch | Every other epoch |

The sketch cGAN is the more complex and ambitious model — it must generate from near-scratch rather than just filling in a localized region.

---

## 3. Dataset

### 3.1 Directory Structure

```
/cgan/
├── train/
│   ├── source/   ← Augmented clock photos (conditioning reference)
│   ├── sketch/   ← Sketch or edge-map representation of the clock
│   └── target/   ← Ground-truth photorealistic clock image
└── val/
    ├── source/
    ├── sketch/
    └── target/
```

Each filename is the same across all three subfolders for a given sample. The dataset includes both a training set and a validation set — this is an important difference from the inpainting project, which only used a training split.

### 3.2 ClockDataset Class

```python
class ClockDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        ...
    def __getitem__(self, idx):
        return src_img, skc_img, tgt_img
```

Each sample returns a triplet:

| Return value | Content | Channels |
|---|---|---|
| `src_img` | Augmented source clock image | 3 (RGB) |
| `skc_img` | Sketch / edge map of the clock | 3 (RGB, treated as a 3-channel image) |
| `tgt_img` | Ground-truth target clock image | 3 (RGB) |

**Difference from the inpainting dataset:**  
Unlike the inpainting model, there is no binary mask in this dataset. Instead, the sketch provides structural guidance. The sketch is loaded and transformed as a standard RGB image — its white/dark line structure serves as a soft structural prior rather than a hard binary locator.

**Note on image loading:** The `.convert('RGB')` calls are commented out in the code, meaning images are loaded in whatever mode they were saved (typically RGB for `.png` and `.jpg`). This is acceptable when the dataset is known to be consistent, but could be a source of bugs if any images were inadvertently saved as grayscale or RGBA.

**System file filtering:**  
```python
self.files = [f for f in self.files if f.endswith('.png') or f.endswith('.jpg')]
```
This filters out macOS `.DS_Store` files and any other non-image system files from the file list, ensuring only valid image files are included.

### 3.3 Data Preprocessing & Transforms

```python
transforms_ = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Three transformations are applied to all images (source, sketch, and target):

1. **Resize to 256×256:** All images are resized to a standard resolution. This is essential because neural networks require fixed-size inputs and all three images must be spatially aligned.

2. **ToTensor:** Converts PIL images with pixel values in `[0, 255]` to PyTorch tensors with values in `[0, 1]`, rearranging from `(H, W, C)` to `(C, H, W)`.

3. **Normalize with mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5):** Shifts values from `[0, 1]` to `[-1, 1]` per channel:
   ```
   output = (input - 0.5) / 0.5
   ```
   This three-value tuple (one per RGB channel) is applied independently to each channel. All three channels use the same normalization values here (0.5/0.5), which is equivalent to the inpainting model's single-value normalization but written explicitly for a 3-channel image.

   Note: Unlike the inpainting model, **no special mask renormalization is needed here** — the sketch is treated like any other image and remains normalized at `[-1, 1]`.

### 3.4 DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
```

Two DataLoaders are created:

- **`train_loader`:** Shuffles batches each epoch (prevents ordering biases) with `batch_size=8`.
- **`val_loader`:** Does not shuffle (`shuffle=False`) with `batch_size=4`. Using a smaller batch for validation is common since validation only requires a forward pass (no gradients stored), so memory usage is lower and even a small subset is enough to gauge quality.

The validation loader is used purely for visualization during training — checking how well the model generalizes to unseen examples after each epoch.

---

## 4. Model Architecture

### 4.1 Generator — Deep U-Net

```python
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
```

The generator is an **8-level deep U-Net**, significantly deeper than the 4-level U-Net in the inpainting model. This extra depth is necessary because the synthesis task (generating a full image from a sketch) requires capturing far more abstract semantic understanding than the localized inpainting task.

**Input:** 6 channels — the source image (3 RGB channels) concatenated with the sketch image (3 RGB channels): `(Batch, 6, 256, 256)`.  
**Output:** 3 channels — a photorealistic clock image: `(Batch, 3, 256, 256)`.

#### Why 6 Input Channels?

The generator is conditioned on two images simultaneously:
- The **source** tells the model about visual style, color palette, clock design, and texture.
- The **sketch** tells the model about the structural layout — where hands, numbers, and dial elements are positioned.

By concatenating them channel-wise before the first convolution, the model learns to integrate both types of information from the very start of the encoding process. This is the standard conditioning approach in pix2pix-style frameworks.

#### Full Architecture — Encoder

```
down1:  (6,   256, 256) → (64,   128, 128)   No normalization
down2:  (64,  128, 128) → (128,   64,  64)   InstanceNorm
down3:  (128,  64,  64) → (256,   32,  32)   InstanceNorm
down4:  (256,  32,  32) → (512,   16,  16)   InstanceNorm + Dropout(0.5)
down5:  (512,  16,  16) → (512,    8,   8)   InstanceNorm + Dropout(0.5)
down6:  (512,   8,   8) → (512,    4,   4)   InstanceNorm + Dropout(0.5)
down7:  (512,   4,   4) → (512,    2,   2)   InstanceNorm + Dropout(0.5)
down8:  (512,   2,   2) → (512,    1,   1)   No normalization + Dropout(0.5)
```

The bottleneck (`down8`) compresses the entire image down to a `1×1` spatial representation with 512 feature channels — a single feature vector that encodes the most abstract, global semantic information about both the source and sketch.

#### Full Architecture — Decoder

```
up1:  (512,    1,   1) → concat with d7 → (1024,  2,   2)
up2:  (1024,   2,   2) → concat with d6 → (1024,  4,   4)
up3:  (1024,   4,   4) → concat with d5 → (1024,  8,   8)
up4:  (1024,   8,   8) → concat with d4 → (1024, 16,  16)
up5:  (1024,  16,  16) → concat with d3 → (512,  32,  32)
up6:  (512,   32,  32) → concat with d2 → (256,  64,  64)
up7:  (256,   64,  64) → concat with d1 → (128, 128, 128)
final:(128,  128, 128) → (3, 256, 256)   Tanh output
```

Each skip connection doubles the channel count at that decoder layer, which is why upsampling blocks are designed with larger input channel counts than output channel counts (e.g., `UNetUp(1024, 512)` takes 1024 channels in and produces 512 channels out, ready to be doubled again by the next skip).

### 4.2 UNetDown and UNetUp Modules

The U-Net is built from two reusable building-block classes, making the architecture cleanly modular.

#### UNetDown

```python
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
```

Each downsampling block:
- **Strided Conv2d** (kernel=4, stride=2, padding=1): Halves spatial dimensions, learns to extract relevant features. Using a learnable convolution for downsampling (instead of fixed pooling) allows the model to decide which spatial information to retain.
- **InstanceNorm2d** (optional): Normalizes activations per sample. Skipped on `down1` (input is already normalized) and `down8` (the bottleneck).
- **LeakyReLU(0.2)**: Prevents dead neurons by allowing a small gradient for negative values.
- **Dropout** (optional, rate=0.5): Randomly zeroes half of the activations during training. Used on the deeper layers to prevent overfitting and encourage robustness.

#### UNetUp

```python
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout: layers.append(nn.Dropout(dropout))

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # Skip Connection applied HERE
        return x
```

Each upsampling block:
- **ConvTranspose2d** (kernel=4, stride=2, padding=1): Doubles spatial dimensions via learned transposed convolution.
- **InstanceNorm2d**: Applied to every upsampling block (unlike the downsampling path, there are no exceptions in the decoder).
- **ReLU**: Standard activation. ReLU (not LeakyReLU) is used in the decoder — this is common practice; the encoder uses LeakyReLU to avoid dying gradients during learning, while the decoder uses standard ReLU since it operates on already-learned, well-distributed feature maps.
- **Dropout** (optional): Applied to the first four upsampling blocks, mirroring the dropout in the corresponding encoder layers.

**Key design — skip connection inside `forward()`:**  
Unlike the inpainting generator where skip connections were handled in the outer `GeneratorUNet.forward()`, here the skip concatenation is done *inside* `UNetUp.forward()`. This encapsulates the skip logic within the block itself and makes the outer forward pass cleaner:

```python
# Inside GeneratorUNet.forward():
u1 = self.up1(d8, d7)   # up1 internally: upsample d8, then cat with d7
u2 = self.up2(u1, d6)   # up2 internally: upsample u1, then cat with d6
...
```

This is an architectural style choice — functionally equivalent to the inpainting generator's approach, but more modular.

### 4.3 The Role of Dropout in the Generator

Dropout is applied at rate `0.5` to the deeper encoder layers (`down4` through `down8`) and the first four decoder layers (`up1` through `up4`). This has two important effects in a GAN context:

**1. Regularization:** The model has millions of parameters and could easily memorize training samples. Dropout forces each neuron to learn robust, generalizable features rather than co-adapting with specific other neurons.

**2. Stochastic generation (optional):** In pix2pix-style GANs, dropout can be left active during inference to introduce controlled stochasticity — running the same input multiple times will produce slightly different outputs. This models the "one-to-many" nature of the task (a single sketch could correspond to many valid realizations). However, whether this model keeps dropout active at inference depends on whether `generator.eval()` is called during inference (which would disable dropout) or if the model is kept in training mode.

### 4.4 Discriminator — Deep PatchGAN

```python
class Discriminator(nn.Module):
    def __init__(self, in_channels=9):  # 3 source + 3 sketch + 3 target = 9
```

The discriminator is a deeper PatchGAN than in the inpainting model — it has **4 convolutional blocks** instead of 3, allowing it to reason about larger receptive fields and more complex spatial patterns.

**Input:** All three images concatenated: `(source, sketch, target)` along the channel dimension → `(Batch, 9, H, W)`.  
**Output:** A spatial patch-level map of real/fake scores.

#### Architecture

```
Block 1: (9,   H,    W   ) → (64,  H/2,  W/2 )   No normalization
Block 2: (64,  H/2,  W/2 ) → (128, H/4,  W/4 )   InstanceNorm
Block 3: (128, H/4,  W/4 ) → (256, H/8,  W/8 )   InstanceNorm
Block 4: (256, H/8,  W/8 ) → (512, H/16, W/16)   InstanceNorm
ZeroPad + Conv2d(512, 1, 4) → (1, ~H/16, ~W/16)
```

For a 256×256 input, the output map is approximately `16×16` — each of the 256 patch-level predictions covers a receptive field corresponding to a `16×16` region of the input image.

**Why does the discriminator see all three inputs?**  
The discriminator must judge whether the generated image is a realistic and *contextually correct* rendering of both the source and the sketch. If it only saw the target image, it could not verify that the output is consistent with the sketch's hand positions or the source's color palette. By seeing all three, it can detect:
- Inconsistencies between the sketch structure and the generated image (e.g., hands in wrong positions).
- Inconsistencies between the source style and the generated image (e.g., wrong color or texture).
- Whether the generated image is photorealistic overall.

The discriminator's `forward()` explicitly names its arguments `img_A` (source), `img_B` (sketch), `img_C` (target), making the conditioning logic transparent:

```python
def forward(self, img_A, img_B, img_C):
    img_input = torch.cat((img_A, img_B, img_C), 1)
    return self.model(img_input)
```

---

## 5. Loss Functions

### 5.1 GAN Loss (Adversarial Loss)

```python
criterion_GAN = nn.MSELoss()  # LSGAN Loss
```

The adversarial loss uses **MSE (Mean Squared Error)** — the Least Squares GAN (LSGAN) formulation. As in the inpainting model, this is more stable than the original binary cross-entropy because it provides a smooth, continuous gradient signal even when the discriminator is very confident.

**Target labels for LSGAN:**
- "Real" label = `1` (a tensor of ones)
- "Fake" label = `0` (a tensor of zeros)

Unlike the inpainting model where target tensors were created dynamically with `torch.ones_like(pred_fake)`, here they are created explicitly with fixed spatial dimensions matching the PatchGAN output:

```python
valid = torch.ones((real_src.size(0), 1, 16, 16), requires_grad=False).to(device)
fake  = torch.zeros((real_src.size(0), 1, 16, 16), requires_grad=False).to(device)
```

This hardcodes the expected output size of `(Batch, 1, 16, 16)`. The advantage of explicit sizes is clarity and slight efficiency (avoids a forward pass just to determine the shape); the disadvantage is that this breaks if the image size or discriminator architecture changes. The `requires_grad=False` prevents PyTorch from tracking gradients through these constant target tensors.

### 5.2 Pixel-wise Loss (L1)

```python
criterion_pixelwise = nn.L1Loss()
loss_pixel = criterion_pixelwise(fake_tgt, real_tgt)
```

A standard, **uniform L1 loss** across all pixels — the mean absolute difference between the generated image and the real target, with equal weight on every pixel.

This differs from the inpainting model's **weighted L1 loss**. In the inpainting model, the hand region (the area being filled) received 50× more penalty than the background. Here, no spatial weighting is applied because the entire image is being generated — there is no localized region of particular importance. The model is equally responsible for getting every pixel right.

### 5.3 Total Generator Loss

```python
loss_G = loss_GAN + (L1_LAMBDA * loss_pixel)
```

The generator's total loss combines adversarial loss and pixel loss:

- **`L1_LAMBDA = 1000`:** Double the inpainting model's `LAMBDA_L1 = 500`. The higher weight on pixel accuracy makes sense here — generating from scratch has more degrees of freedom than inpainting, so a stronger pixel-level anchor is needed to prevent the generator from drifting toward generating arbitrary but photorealistic clocks that don't match the sketch.

### 5.4 Discriminator Loss

```python
loss_real = criterion_GAN(pred_real, valid)
loss_fake = criterion_GAN(pred_fake, fake)
loss_D = 0.5 * (loss_real + loss_fake)
```

Standard adversarial discriminator loss: train D to output `1` for real `(source, sketch, target)` triplets and `0` for `(source, sketch, fake_target)` triplets, averaged with a `0.5` scaling factor.

As in the inpainting model, `fake_tgt.detach()` is used when computing `loss_fake` — this is critical to prevent the discriminator's backward pass from modifying the generator's weights.

---

## 6. Weight Initialization

```python
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
```

Identical to the inpainting model — all convolutional weights are initialized from a Normal distribution with mean=0 and std=0.02 (the standard DCGAN initialization). This small-variance initialization keeps early activations well-behaved and gradients stable at the start of training.

The BatchNorm2d branch of this function is carried over for compatibility, even though this model uses **InstanceNorm2d** throughout. Since no `BatchNorm2d` layers exist in the model, this branch never executes — it's harmless dead code.

---

## 7. Optimizers

```python
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

Both networks use the **Adam optimizer** with the same hyperparameters as the inpainting model:
- **`lr=0.0002`:** The classic GAN learning rate from DCGAN.
- **`betas=(0.5, 0.999)`:** `beta1=0.5` reduces momentum for GAN stability (faster adaptation to changing gradient directions); `beta2=0.999` keeps per-parameter learning rate adaptation stable.

---

## 8. Training Loop

### 8.1 Step-by-Step Walkthrough

For each epoch and each batch:

1. Load `(source, sketch, target)` from the training DataLoader.
2. Concatenate source and sketch → `gen_input` of shape `(Batch, 6, 256, 256)`.
3. **Always train the Generator** (every batch).
4. **Train the Discriminator every other epoch** (only on even epochs).
5. Log losses every 100 batches.
6. After each epoch: visualize on the validation set.
7. Every 10 epochs: save the generator checkpoint.

### 8.2 Generator Training Step

```python
optimizer_G.zero_grad()

gen_input = torch.cat((real_src, real_skc), 1)   # 6-channel input
fake_tgt = generator(gen_input)                   # Generate fake target

pred_fake = discriminator(real_src, real_skc, fake_tgt)
loss_GAN = criterion_GAN(pred_fake, valid)        # Adversarial loss

loss_pixel = criterion_pixelwise(fake_tgt, real_tgt)  # Pixel loss

loss_G = loss_GAN + (L1_LAMBDA * loss_pixel)
loss_G.backward()
optimizer_G.step()
```

The generator is updated every single batch. The gradient flows through both the pixel loss (directly comparing `fake_tgt` to `real_tgt`) and the GAN loss (which flows through the discriminator's forward pass back to the generator). Together these two signals pull the generator toward outputs that are both accurate and photorealistic.

### 8.3 Discriminator Training Step

```python
if epoch % 2 == 0:
    optimizer_D.zero_grad()

    pred_real = discriminator(real_src, real_skc, real_tgt)
    loss_real = criterion_GAN(pred_real, valid)

    pred_fake = discriminator(real_src, real_skc, fake_tgt.detach())
    loss_fake = criterion_GAN(pred_fake, fake)

    loss_D = 0.5 * (loss_real + loss_fake)
    loss_D.backward()
    optimizer_D.step()
```

### 8.4 Asymmetric Update Schedule

This is the most notable structural difference from the inpainting training loop. The discriminator is only updated **on even epochs** (epoch 0, 2, 4, 6, ...) — meaning the generator is trained twice as often as the discriminator across the full training run.

**Why update the discriminator less frequently?**

In GAN training, a fundamental instability occurs when the discriminator becomes too powerful too quickly. If D can perfectly distinguish real from fake early in training, the generator receives gradients of near-zero magnitude (the loss saturates), and learning stalls. By updating D every other epoch while G updates every batch, this approach deliberately keeps the discriminator from getting too far ahead of the generator.

This is a training stabilization strategy tailored to this task. The generation problem (sketch to full image) is harder than inpainting, so the generator needs more "breathing room" to develop before facing a highly competent discriminator. The asymmetry ensures G always gets a useful learning signal from D, even in the early stages of training.

---

## 9. Hyperparameters Summary

| Parameter | Value | Meaning |
|---|---|---|
| `BATCH_SIZE` | 8 | Samples per training step |
| `LEARNING_RATE` | 0.0002 | Adam learning rate for both G and D |
| `EPOCHS` | 101 | Total training epochs |
| `IMG_SIZE` | 256 | Input/output resolution (pixels) |
| `L1_LAMBDA` | 1000 | Pixel loss weight (2× the inpainting model) |
| Generator input channels | 6 | Source (3) + Sketch (3) |
| Discriminator input channels | 9 | Source (3) + Sketch (3) + Target (3) |
| U-Net depth | 8 levels | Down8 → Up7 + final |
| Dropout rate (deep layers) | 0.5 | Applied to down4–down8, up1–up4 |
| PatchGAN output size | 16×16 | Each cell evaluates a patch of the input |
| Discriminator update frequency | Every 2 epochs | To prevent discriminator from overpowering generator |
| Adam `beta1` | 0.5 | GAN-stable momentum |
| Adam `beta2` | 0.999 | Standard adaptive LR |
| Weight init std | 0.02 | Normal distribution, DCGAN standard |

---

## 10. Visualization & Checkpointing

### Visualization

After every epoch, `sample_images` is called on the **validation loader** (not the training loader):

```python
sample_images(val_loader, generator, epoch, i)
```

This is an important difference from the inpainting model, which visualized training data. Using validation data gives a more honest view of generalization — it shows whether the model performs on images it has never trained on, making it easier to spot overfitting early.

The function displays four panels:

| Panel | Content |
|---|---|
| Source (Augmented) | The source clock image used as style reference |
| Sketch | The structural sketch used as layout guide |
| Generated | The model's synthesized output |
| Real Target | The ground-truth target image |

Images are denormalized from `[-1, 1]` back to `[0, 1]` for display:
```python
img_np = (img_np * 0.5) + 0.5
```

Note: Unlike the inpainting model, `generator.eval()` and `generator.train()` are **not** called in `sample_images`. This means dropout remains active during visualization, introducing slight randomness into the displayed output. This may be intentional (to show the model's stochastic behavior) or an oversight.

### Checkpointing

```python
if epoch % 10 == 0:
    torch.save(generator.state_dict(), f"generator_{epoch}.pth")
```

The generator's weights are saved every 10 epochs. Only the generator is saved (not the discriminator), since at inference time only the generator is needed. The `state_dict()` approach saves just the parameters as a plain dictionary, making the checkpoint portable and independent of the class definition.

---

## 11. Comparison to the Inpainting GAN

| Feature | Inpainting GAN | Sketch cGAN |
|---|---|---|
| **Task** | Erase clock hands | Synthesize clock from sketch |
| **Generator input** | Source + binary mask (4ch) | Source + sketch (6ch) |
| **Discriminator input** | Source+mask + target (7ch) | Source + sketch + target (9ch) |
| **U-Net depth** | 4 encoder + 4 decoder | 8 encoder + 7 decoder + final |
| **Dropout** | None | Heavy (0.5) on deep layers |
| **Pixel loss** | Weighted L1 (50× mask region) | Uniform L1 |
| **L1 weight** | 500 | 1000 |
| **Discriminator update** | Every batch | Every other epoch |
| **Validation split** | No | Yes |
| **Visualization data** | Train | Val |
| **PatchGAN depth** | 3 conv blocks | 4 conv blocks |
| **Mask binarization** | Yes (hard threshold at 0.5) | N/A |

---

## 12. Key Design Decisions Explained

### Why 8-Level U-Net Instead of 4?

The sketch synthesis task requires much deeper semantic understanding than localized inpainting. The 8-level U-Net compresses a 256×256 image all the way to a `1×1` bottleneck — a single global feature vector. This forces the model to develop a holistic, abstract understanding of the clock as a whole (global structure, hand configuration, overall layout) before gradually reconstructing spatial details during decoding. With only 4 levels, the bottleneck would be `16×16`, which retains too much local spatial information and may lead to the model relying on low-level pattern matching rather than genuine semantic understanding.

### Why Uniform L1 (No Spatial Weighting)?

In the inpainting task, there was a well-defined "hard region" (the hand pixels) that the model needed to focus on — hence the 50× weighting. In the sketch synthesis task, the entire image must be generated correctly. There is no sub-region that is inherently harder or more important than the rest, so a uniform pixel loss is appropriate. The high base value of `L1_LAMBDA=1000` provides strong overall pixel accuracy pressure across the whole image.

### Why L1_LAMBDA = 1000 (Higher Than Inpainting's 500)?

Generating a full image from a sketch has far more degrees of freedom than filling in a masked region. The generator could potentially learn to produce clock images that are photorealistic but don't closely match the target. A higher pixel loss weight (1000 vs. 500) provides a stronger constraint, keeping the generated images tethered more tightly to the ground truth. This is especially important early in training before the GAN dynamics have stabilized.

### Why Update the Discriminator Every Other Epoch?

This is a deliberate training stabilization choice for a harder task. The synthesis task is more demanding — the generator has to produce full images from scratch. If the discriminator gets too far ahead of the generator (which can happen quickly since it only needs to classify, not generate), the generator's adversarial gradient signal becomes uninformative (near-zero or saturated). By training D only half as often, the generator is given more updates to develop competence before facing a strong discriminator.

### Why Does the Discriminator Accept Three Separate Images?

The discriminator's `forward(img_A, img_B, img_C)` signature is a cleaner API than the inpainting model's `forward(img_input, img_target)`. Internally both approaches concatenate along the channel dimension before the first layer, so the distinction is purely stylistic — but naming the three inputs explicitly (`source`, `sketch`, `target`) makes the code more readable and self-documenting.

### Why Is the Sketch Treated as a 3-Channel Image?

Sketches are often grayscale by nature, but by treating them as 3-channel RGB images, no special handling is needed in the pipeline. The three channels of the sketch image would all carry the same grayscale information. This keeps the code simple and unified — all images go through the same transform pipeline — at the cost of slightly more memory (3× channels instead of 1×). A more memory-efficient design would use a 1-channel grayscale sketch, resulting in a 4-channel generator input (3 source + 1 sketch), but that would require separating the transform pipeline.

### Why Use Validation Data for Visualization?

Monitoring performance on the validation set (rather than training data) provides an honest assessment of generalization. If the model's training-set outputs look great but validation outputs look poor, that's an early warning sign of overfitting. Seeing validation results in real-time during training allows for early stopping or learning rate adjustments before too much compute is wasted.