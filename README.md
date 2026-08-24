 Polyp Segmentation in Multimodal Endoscopic Images Using ResUNet++

A deep-learning notebook for automatic polyp segmentation in endoscopic images from the PolypDB dataset. The project trains a separate ResUNet++ model for each imaging modality and produces a binary mask identifying the pixels belonging to a polyp.

## Overview

Polyp segmentation is the task of locating a polyp precisely within an endoscopic image. Unlike image classification, which predicts whether a polyp is present, segmentation predicts a mask for every pixel.

This notebook supports the following PolypDB modalities:

- **NBI**: Narrow Band Imaging
- **LCI**: Linked Color Imaging
- **FICE**: Flexible Spectral Imaging Color Enhancement
- **BLI**: Blue Light Imaging

Each modality provides a different view of mucosal and vascular structures. Training modality-specific models allows their performance to be compared under different imaging conditions.

## Model

The notebook implements a ResUNet++-style encoder-decoder network in PyTorch. Its main components are:

- Residual convolutional blocks
- Squeeze-and-Excitation channel attention
- Atrous Spatial Pyramid Pooling (ASPP)
- Attention-based decoder blocks
- A one-channel output layer for binary segmentation

The model returns logits. A sigmoid activation is applied during evaluation to obtain probabilities, followed by a threshold of `0.5` to create the final binary mask.

## Notebook Workflow

The notebook is organized into the following stages:

1. **Visualize samples**
   - Loads images and masks from each modality.
   - Displays randomly selected image-mask pairs.
   - Checks that image and mask filenames match.

2. **Augment the dataset**
   - Saves the original images and masks.
   - Generates ten augmented versions per image.
   - Uses horizontal and vertical flips, rotation, brightness/contrast changes, Gaussian blur, and shift-scale-rotation transforms.
   - Resizes saved data to `512 x 512`.

3. **Build the ResUNet++ model**
   - Defines the network architecture.
   - Optionally reports parameter count and FLOPs using `ptflops`.

4. **Define losses and metrics**
   - Dice loss
   - Dice plus binary cross-entropy loss
   - Jaccard/IoU
   - Dice/F1 score
   - Recall
   - Precision
   - Accuracy
   - F2 score
   - Hausdorff distance helper

5. **Train models**
   - Splits each modality into training, validation, and test sets.
   - Trains one model per modality.
   - Saves the best checkpoint according to validation F1 score.
   - Uses learning-rate reduction and early stopping.

6. **Test models**
   - Loads each modality-specific checkpoint.
   - Predicts masks for the test set.
   - Calculates evaluation metrics and mean FPS.
   - Saves predicted masks and side-by-side comparison images.

7. **Display results**
   - Displays the original image, ground-truth mask, and predicted mask for randomly selected test examples.

## Requirements

Recommended environment:

- Python 3.9 or newer
- PyTorch
- CUDA-capable GPU recommended
- Jupyter Notebook or VS Code with the Jupyter extension

Install the main dependencies with:

```bash
pip install numpy opencv-python pillow matplotlib imageio tqdm scikit-learn scipy albumentations torch torchvision ptflops
```

The notebook currently installs `ptflops` in one of its cells. The other dependencies should be installed before running the notebook if they are not already available.

## Dataset Structure

The notebook expects the modality-wise PolypDB layout to look like this:

```text
PolypDB_modality_wise/
├── BLI/
│   ├── images/
│   └── masks/
├── FICE/
│   ├── images/
│   └── masks/
├── LCI/
│   ├── images/
│   └── masks/
└── NBI/
    ├── images/
    └── masks/
```

Image and mask files should have matching filename stems. For example:

```text
images/example_001.jpg
masks/example_001.png
```

Both `.jpg` and `.png` files are supported by the loading code.

## Running the Notebook

Run the cells in order:

1. Visualize the dataset.
2. Generate augmented data.
3. Define the ResUNet++ model.
4. Define losses, metrics, and utility functions.
5. Train the models.
6. Test the saved models.
7. Display the prediction results.

The notebook was originally configured for Kaggle. Before running it elsewhere, update the dataset and output paths from `/kaggle/input` and `/kaggle/working` to local paths.

A typical local path configuration could be:

```python
base_path = "./data/PolypDB/PolypDB_modality_wise"
augmented_data_path = "./outputs/augmented_data/NBI"
results_path = "./outputs/results/NBI"
```

## Training Configuration

The current training cell uses:

| Setting | Value |
|---|---:|
| Input size | `256 x 256` |
| Batch size | `8` |
| Maximum epochs | `80` |
| Optimizer | Adam |
| Learning rate | `0.0001` |
| Loss | Dice + Binary Cross-Entropy |
| Early-stopping patience | `50` epochs |
| Random seed | `42` |

The train/validation/test split is approximately 80%/8%/12%.

## Generated Outputs

Training creates a checkpoint for each modality:

```text
files/
├── BLI_checkpoint.pth
├── FICE_checkpoint.pth
├── LCI_checkpoint.pth
├── NBI_checkpoint.pth
└── train_log.txt
```

Testing creates results similar to:

```text
results/
└── NBI/
    ├── joint/
    │   └── comparison_images.png
    └── mask/
        └── predicted_masks.png
```

A joint comparison image contains:

1. Original endoscopic image
2. Ground-truth mask
3. Model prediction

## Evaluation Metrics

- **Jaccard/IoU**: Measures overlap between the predicted and true masks.
- **Dice/F1**: Measures segmentation overlap and is useful when the polyp occupies a small part of the image.
- **Precision**: Measures how much of the predicted polyp area is correct.
- **Recall**: Measures how much of the actual polyp area was detected.
- **F2 score**: Gives more weight to recall than precision.
- **Hausdorff distance**: Measures the largest boundary discrepancy between prediction and ground truth.
- **FPS**: Estimates inference speed.

Higher Dice, Jaccard, precision, recall, and F2 scores are generally better. Lower Hausdorff distance is better.
