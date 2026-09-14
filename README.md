# Cloud Base Height Classification

This repository contains a PyTorch classification pipeline for assigning sky images to cloud-base height categories. It supports training and inference with configurable convolutional and transformer-based backbones, validation, checkpointing, and experiment tracking with Weights & Biases (W&B).

## Class definition

The dataset converts cloud-base height values into discrete classes:

- Class 0: height < 2000 m
- Class 1: 2000 m <= height < 6000 m
- Class 2: height >= 6000 m

The class assignment logic is implemented in [dataset/dataset.py](dataset/dataset.py).

## Repository Structure

The repository includes the main training and inference pipeline, dataset implementation, model architectures, and generated evaluation results.

```text
.
├── archs/                          # Model architectures
├── baseline.yml                   # Main training configuration
├── dataset/                       # Dataset implementation and label logic
├── datos/                         # Image-to-label split files (.txt)
├── ejecutar.txt                   # Local execution notes and command shortcuts
├── imagenes_train/                # Training images
├── imagenes_val/                  # Validation images
├── imagenes_test/                 # Test images
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── results/                       # Trained models and inference outputs
├── run.py                         # Main training and inference entry point
├── scripts/                       # Training, validation, and testing scripts
├── utils/                         # Augmentations, losses, and utilities
├── venv_ALTURA/                   # Local virtual environment
└── wandb/                         # Local W&B artifacts and run metadata
```

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or newer recommended
- An NVIDIA GPU with CUDA is recommended for training
- A Weights & Biases account and API key when W&B logging is enabled

## Installation

Create and activate a virtual environment from the repository root.

### Windows Command Prompt

```bat
python -m venv venv_ALTURA
venv_ALTURA\Scripts\activate
```

### Windows PowerShell

```powershell
python -m venv venv_ALTURA
.\venv_ALTURA\Scripts\Activate.ps1
```

Install PyTorch with the CUDA 11.8 wheels used by the original setup:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages:

```bash
pip install timm wandb fvcore PyYAML tqdm opencv-python matplotlib imageio pillow numpy ipykernel
```

The pinned dependency file is also available for reproducing the recorded environment:

```bash
pip install -r requirements.txt
```

Use either the explicit installation commands above or the requirements file according to your local CUDA/PyTorch setup. If CUDA is not available, install a CPU-compatible PyTorch build from the official PyTorch installation selector and install the remaining packages afterward.

## Weights & Biases Setup

The baseline configuration enables W&B with `wandb.use: True`. Authenticate with your own API key before training:

```bash
wandb login
```

When prompted, paste the API key from your W&B account. Alternatively, set it as an environment variable.

### Windows Command Prompt

```bat
set WANDB_API_KEY=YOUR_WANDB_API_KEY
```

### Windows PowerShell

```powershell
$env:WANDB_API_KEY = "YOUR_WANDB_API_KEY"
```

Do not commit API keys to the repository. If W&B is not required, set `wandb.use: False` in `baseline.yml`.

## Image Folders and Dataset Layout

This project currently uses a single training configuration, `baseline.yml`, and expects the following image folders at the repository root:

```text
imagenes_train/
├── ...

imagenes_val/
├── ...

imagenes_test/
├── ...
```

Each folder contains the images for its corresponding split. The files in `datos/` must match those images and follow this format:

```text
image_filename.jpg;cloud_base_height
```

Example:

```text
C009_20240224_2034.jpg;7957
C009_20240601_1025.jpg;1495
```

The `CloudDataset` loader expects one sample per line, and the cloud-base height is normalized internally by dividing it by `10000` during training.

The `CloudDataset` loader expects one sample per line, and converts the cloud-base height into one of the three classes during loading.

The current repository uses these split files:

```text
datos/train.txt
datos/train_day.txt
datos/train_night.txt
datos/val.txt
datos/val_day.txt
datos/val_night.txt
datos/test.txt
datos/test_day.txt
datos/test_night.txt
```

Create or replace them with your own data so that every image name listed there exists in the corresponding folder and the numerical height is in metres.


## Training

From the repository root, run:

```bash
python run.py --mode train --config baseline.yml
```

Training writes model weights and related outputs to `results/`. Checkpoints used to resume training are stored in `checkpoints/`.

You can select the CUDA device with `--device` and provide an experiment name with `--name`:

```bash
python run.py --mode train --config baseline.yml --device 0 --name my-model
```

## Inference

Inference expects a trained model at `results/<model-name>/<model-name>.pt`.

```bash
python run.py --mode inference --config baseline.yml --name test-model_082337
```

Inference results are written to the relevant results directory, including prediction and error files when enabled in the YAML configuration.

## Configuration

`baseline.yml` controls dataset directories, batch sizes, training epochs, augmentations, optimizer settings, model selection, W&B logging, and output behavior. The `model.pick` value selects one of the models listed under `model.models`.

You should keep the YAML values consistent with your dataset folders and split files, especially:

```yaml
train:
  train_dir: '.\imagenes_train'

validation:
  val_dir: '.\imagenes_val'

test:
  test_dir: '.\imagenes_test'
```

## Notes

- Run commands from the repository root so relative paths resolve correctly.
- Ensure image names in every `.txt` file exactly match the corresponding image files.
- Keep W&B credentials outside version control.
- Large datasets, model weights, W&B runs, and generated results are normally better stored outside the Git repository or managed with Git LFS.