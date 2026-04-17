# Anime Face Generation using DCGAN

A Deep Convolutional Generative Adversarial Network (DCGAN) trained from scratch on
over 63,000 anime face images using PyTorch. The Generator learns to produce realistic
64x64 anime faces from random noise vectors, while the Discriminator learns to distinguish
real images from generated ones. A timelapse video captures the generator's improvement
across all 100 epochs.

## How It Works

A GAN consists of two networks trained in opposition:

- Generator — takes a random latent vector (128-dim) and upsamples it through transposed
  convolutions to produce a 64x64 RGB image
- Discriminator — takes a 64x64 image and classifies it as real or fake using strided
  convolutions down to a single sigmoid output

The two networks compete: the Generator tries to fool the Discriminator, and the
Discriminator tries not to be fooled. Over time, the Generator produces increasingly
convincing anime faces.

## Dataset

- Source: Anime Face Dataset on Kaggle
- URL: https://www.kaggle.com/splcher/animefacedataset
- Size: ~63,000 anime face images
- Image size: 64 x 64 pixels (resized and centre-cropped)
- Normalisation: mean (0.5, 0.5, 0.5), std (0.5, 0.5, 0.5)

The dataset is downloaded automatically using the opendatasets library with your Kaggle credentials.

## Model Architectures

### Generator

```
Input: latent vector (128 x 1 x 1)
        |
ConvTranspose2d  512 filters, 4x4  →  (512 x 4 x 4)   + BatchNorm + ReLU
        |
ConvTranspose2d  256 filters, 4x4  →  (256 x 8 x 8)   + BatchNorm + ReLU
        |
ConvTranspose2d  128 filters, 4x4  →  (128 x 16 x 16) + BatchNorm + ReLU
        |
ConvTranspose2d   64 filters, 4x4  →  (64 x 32 x 32)  + BatchNorm + ReLU
        |
ConvTranspose2d    3 filters, 4x4  →  (3 x 64 x 64)   + Tanh
        |
Output: 64 x 64 RGB image
```

### Discriminator

```
Input: (3 x 64 x 64)
        |
Conv2d  64 filters, stride 2  →  (64 x 32 x 32)  + BatchNorm + LeakyReLU(0.2)
        |
Conv2d 128 filters, stride 2  →  (128 x 16 x 16) + BatchNorm + LeakyReLU(0.2)
        |
Conv2d 256 filters, stride 2  →  (256 x 8 x 8)   + BatchNorm + LeakyReLU(0.2)
        |
Conv2d 512 filters, stride 2  →  (512 x 4 x 4)   + BatchNorm + LeakyReLU(0.2)
        |
Conv2d   1 filter,  stride 1  →  (1 x 1 x 1)     + Flatten + Sigmoid
        |
Output: real/fake probability
```

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Latent size | 128 |
| Image size | 64 x 64 |
| Batch size | 128 |
| Epochs | 100 |
| Learning rate | 0.001 |
| Optimiser | Adam (betas: 0.5, 0.999) |
| Loss function | Binary Cross-Entropy |
| Device | CUDA (GPU) |

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | Model building and training |
| torchvision | Dataset loading, transforms, image grid utilities |
| opendatasets | Kaggle dataset download |
| OpenCV (cv2) | Timelapse video creation |
| matplotlib | Loss and score plotting |
| tqdm | Training progress bars |
