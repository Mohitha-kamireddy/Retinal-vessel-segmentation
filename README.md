# Retinal Vessel Segmentation (DRIVE Dataset)

A U-Net implemented in PyTorch for retinal blood vessel segmentation, trained and evaluated on the DRIVE dataset.

## Overview

- **Architecture:** standard U-Net (encoder-decoder with skip connections), single-channel binary output
- **Dataset:** DRIVE (Digital Retinal Images for Vessel Extraction), 20 training images, split 80/20 into train/validation
- **Loss:** BCE + Dice
- **Optimizer:** Adam
- **Augmentation:** horizontal/vertical flip, 90° rotation, affine (shift/scale/rotate), brightness/contrast jitter (via Albumentations)
- **Training:** 200 epochs, batch size 4, learning rate 1e-4

## Results

Evaluated on a held-out validation split (4 images) from the DRIVE training set, with metrics restricted to the field-of-view (FOV) mask:

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9272 |
| Precision | 0.6505 |
| Recall    | 0.8312 |
| Dice      | 0.7298 |
| IoU       | 0.5746 |

Sample prediction (Retinal Image | Ground Truth | Prediction):

![Sample prediction](outputs/predictions/prediction_0.png)

The model captures the overall vascular structure well, including major vessels and branching patterns. Recall is notably higher than precision, indicating the model tends to over-predict vessel pixels (visible as speckle noise in the prediction masks) rather than missing them.

## Project Structure

```
retinal-vessel-segmentation/
├── data/
│   ├── raw/DRIVE/        # DRIVE dataset (training/test, images/masks/FOV)
│   └── processed/        # Preprocessed data
├── src/
│   ├── dataset.py             # PyTorch Dataset for DRIVE
│   ├── preprocessing.py       # Green-channel extraction, CLAHE, normalization, resizing
│   ├── transforms.py          # Albumentations augmentation pipelines
│   ├── model.py                # U-Net architecture
│   ├── train.py                 # Training loop with checkpointing
│   ├── evaluate.py              # Accuracy / Precision / Recall / Dice / IoU (FOV-masked)
│   └── visualize_predictions.py # Saves side-by-side prediction images
├── outputs/
│   ├── checkpoints/      # Saved model weights (best_model.pth)
│   ├── predictions/      # Prediction visualizations
│   ├── plots/            # Training curves, metric plots
│   └── metrics.csv       # Evaluation metrics
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Place the DRIVE dataset under `data/raw/DRIVE/`, matching this structure:

```
data/raw/DRIVE/
├── training/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
└── test/
    ├── images/
    └── mask/
```

## Usage

Train:
```bash
python3 src/train.py --epochs 200 --batch-size 4 --lr 1e-4
```

Evaluate on the held-out validation split:
```bash
python3 src/evaluate.py
```

Visualize predictions:
```bash
python3 src/visualize_predictions.py
```

## Notes

- The official DRIVE test set ground-truth masks were not consistently available across mirrors, so evaluation uses a held-out validation split from the training set instead.
- Metrics are computed only within the FOV mask, excluding background outside the retina.
