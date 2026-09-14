# Cloud Base Height Classification

This project trains a deep learning model to classify sky images into cloud-base height categories. The implementation in this repository is a 3-class classification task, not a continuous regression problem.

## Class definition

The dataset converts cloud-base height values into discrete classes:

- Class 0: height < 2000 m
- Class 1: 2000 m <= height < 6000 m
- Class 2: height >= 6000 m

The class assignment logic is implemented in [dataset/dataset.py](dataset/dataset.py).

## What the project does

- Loads RGB sky images from train/validation/test folders
- Reads split files with the format `image.jpg;height`
- Converts height values into class labels
- Trains CNN or transformer models for multiclass classification
- Validates performance and saves checkpoints/results
- Logs metrics to Weights & Biases when enabled

## Project structure

```text
.
├── archs/                  # Model architectures
├── dataset/                # Dataset loader and label logic
├── datos/                  # Train/val/test split files (.txt)
├── imagenes_train/         # Training images
├── imagenes_val/           # Validation images
├── imagenes_test/          # Test images
├── results/                # Trained weights and evaluation outputs
├── scripts/                # Training, validation and evaluation code
├── utils/                  # Data augmentation and utilities
├── baseline.yml            # Main experiment configuration
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── run.py                  # Training/inference entry point
├── wandb/                  # W&B local run metadata
├── venv_ALTURA/            # Local virtual environment
└── tea_debug.log           # Debug log file
```

## Environment requirements

- Python 3.10+
- PyTorch and torchvision
- CUDA-enabled GPU recommended for faster training
- Dependencies listed in [requirements.txt](requirements.txt)
- Optional: Weights & Biases account for `wandb.use: True`

## Installation

From the repository root:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Data format

Each line in the dataset text files must follow this pattern:

```text
image_name.jpg;height
```

Example:

```text
C009_20240224_2034.jpg;7957
C009_20240601_1025.jpg;1495
```

The file names must match the corresponding images in the dataset directories.

The split files used in this project are:

```text
datos/train.txt
datos/val.txt
datos/test.txt
datos/train_day.txt
datos/val_day.txt
datos/test_day.txt
datos/train_night.txt
datos/val_night.txt
datos/test_night.txt
```

## Configuration

The main setup is defined in [baseline.yml](baseline.yml). It controls:

- dataset directories
- batch size and workers
- optimizer and scheduler settings
- model choice
- W&B usage
- classification loss

The current project configuration uses 3 output classes and a model selected by `model.pick`:

```yaml
model:
  pick: 8
  models:
    - model: "convnext_v2"
      num_classes: 3
```

This confirms the active task is multi-class classification instead of regression.

## Training

From the repository root:

```bash
python run.py --mode train --config baseline.yml
```

Optional arguments:

```bash
python run.py --mode train --config baseline.yml --device 0 --name my_model
```

Training will save model checkpoints and output artifacts under the [results](results) folder.

## Validation

Validation is run during training automatically via [scripts/val.py](scripts/val.py). The evaluation includes metrics such as:

- accuracy
- loss
- MAE
- RMSE
- standard deviation
- confusion matrix (when enabled)

## Inference

To run inference with a saved model:

```bash
python run.py --mode inference --config baseline.yml --name test-model_110040
```

The model name should match the folder and checkpoint file generated under [results](results).

## Weights & Biases

The project can log training metrics to W&B when `wandb.use` is enabled in the YAML file.

To authenticate:

```bash
wandb login
```

If you want to disable tracking:

```yaml
wandb:
  use: False
```

## Important implementation note

Although some older references may describe a regression task, the current codebase actually performs 3-class cloud-height classification. The evidence is:

- [dataset/dataset.py](dataset/dataset.py): height values are mapped to classes 0, 1, and 2
- [baseline.yml](baseline.yml): `num_classes: 3`
- [scripts/train.py](scripts/train.py): uses `CrossEntropyLoss`
- [scripts/val.py](scripts/val.py): evaluates accuracy/confusion matrix

## Recommended workflow

1. Create the virtual environment and install dependencies.
2. Check that the images and split files are aligned in the expected folders.
3. Update the paths in [baseline.yml](baseline.yml) if needed.
4. Run training with `python run.py --mode train --config baseline.yml`.
5. Check the saved results and checkpoints in [results](results).


