# Cloud Base Height Classification

This repository contains a deep learning pipeline for classifying sky images according to cloud-base height. The actual task implemented by this project is not continuous regression, but a three-class classification based on the estimated cloud height.

In the dataset, the height values are converted into discrete labels:

- Class 0: height < 2000 m
- Class 1: 2000 m <= height < 6000 m
- Class 2: height >= 6000 m

The exact logic is implemented in [dataset/dataset.py](dataset/dataset.py), where each image is assigned a class based on the value found in the data file.

## What this repository does

- Loads sky images from train/val/test folders.
- Reads a `.txt` file with the format `image_name.jpg;height`
- Converts the height into a discrete class for classification training.
- Trains CNN or transformer models on RGB images.
- Evaluates validation results and saves checkpoints and outputs.
- Tracks experiments with Weights & Biases.

## Repository structure

```text
.
├── archs/                  # Available model architectures
├── config/                 # Additional configuration files
├── dataset/                # Dataset implementation
├── datos/                  # Dataset split files (.txt)
├── imagenes_day/           # Daytime images
├── imagenes_night/         # Night-time images
├── imagenes_todoeldia/     # Images from the full day
├── results/                # Trained models and inference outputs
├── scripts/                # Training, validation, and testing logic
├── utils/                  # Augmentations, losses, and utilities
├── baseline.yml            # Full-day training configuration
├── baseline_day.yml        # Daytime training configuration
├── baseline_night.yml      # Night-time training configuration
├── requirements.txt        # Python dependencies
└── run.py                  # Main training and inference entry point
```

## Requirements

- Python 3.10+
- PyTorch + torchvision
- CUDA is recommended for faster training
- Project dependencies are listed in [requirements.txt](requirements.txt)
- A Weights & Biases account if `wandb.use: True` is enabled

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or in Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If you prefer to install manually, the project typically uses:

```bash
pip install torch torchvision torchaudio timm wandb fvcore PyYAML tqdm opencv-python matplotlib imageio pillow numpy scikit-learn seaborn pandas
```

## Data format

Each line in the `.txt` files must follow this pattern:

```text
image.jpg;height
```

Example:

```text
IMG_0001.jpg;1250.5
IMG_0002.png;980.0
IMG_0003.jpg;6500.0
```

The filenames must match the images in the corresponding directory. The data loader checks that the image exists and then converts the height value into a class.

The expected files in this repository are:

```text
datos/train.txt
datos/val.txt
datos/test.txt
```

There are also day/night variants:

```text
datos/train_day.txt
datos/val_day.txt
datos/test_day.txt
datos/train_night.txt
datos/val_night.txt
datos/test_night.txt
```

## Project configuration

The YAML files define image paths, batch size, architecture, learning rate, and W&B usage. The main configurations are:

- [baseline.yml](baseline.yml): full-day dataset
- [baseline_day.yml](baseline_day.yml): daytime images only
- [baseline_night.yml](baseline_night.yml): nighttime images only

In [baseline.yml](baseline.yml), the model is configured with `num_classes: 3` and `model.pick: 8`, for example a ConvNeXt-V2 model. This confirms that the project is working on a multi-class classification task rather than continuous height regression.

## Training

From the repository root:

```bash
python run.py --mode train --config baseline.yml
```

You can also train on the day or night configuration:

```bash
python run.py --mode train --config baseline_day.yml
python run.py --mode train --config baseline_night.yml
```

Optionally, you can select the device and the experiment name:

```bash
python run.py --mode train --config baseline.yml --device 0 --name my_model
```

The results and checkpoints are saved in the [results](results) and [checkpoints](checkpoints) directories when applicable.

## Validation

Validation runs automatically during training through the function in [scripts/val.py](scripts/val.py). This function computes:

- accuracy
- loss
- MAE
- RMSE
- standard deviation
- confusion matrix (if enabled)

## Inference

To run inference with a trained model:

```bash
python run.py --mode inference --config baseline.yml --name test-model_110040
```

The name must match the results folder and the generated `.pt` file.

## Important note about the current implementation

Although the original YAML files and documentation refer to a regression problem, the actual code in this repository performs 3-class classification. The evidence is:

- [dataset/dataset.py](dataset/dataset.py): converts height into classes `0, 1, 2`
- [baseline.yml](baseline.yml): `num_classes: 3`
- [scripts/train.py](scripts/train.py): uses `CrossEntropyLoss`
- [scripts/val.py](scripts/val.py): evaluates accuracy and confusion matrix

For this reason, the README has been adjusted to reflect the actual task performed by the project.

## Weights & Biases

The project can log metrics to W&B when `wandb.use` is enabled in the configuration. To authenticate:

```bash
wandb login
```

If you do not want to use W&B, disable it in the YAML:

```yaml
wandb:
  use: False
```

## Recommendations

- Run the commands from the repository root.
- Make sure the images and `.txt` files remain synchronized.
- Check that the directories used by `train_dir`, `val_dir`, and `test_dir` match the actual dataset structure.
- When switching between day, night, or full-day data, ensure the corresponding split files are also updated.

