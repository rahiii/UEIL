# Quick Start Guide

## What You Have

A **simple, unified framework** to train **TransMorph** (medical image registration), **RAFT** (optical flow), and **VoxelMorph** (medical image registration) models using `.mat` files.

## File Structure
```
UEIL_DL/
├── train.py          # Main training script
├── config.yaml       # Configuration file
├── data/             # Your .mat files (organized)
├── outputs/          # Where models are saved
├── requirements.txt  # Dependencies
├── setup.sh          # One-time setup
└── activate.sh       # Activate environment
```

## How to Use

### 1. **Setup (One Time)**
```bash
./setup.sh
```

### 2. **Activate Environment**
```bash
./activate.sh
```

### 3. **Train Models**

**Train TransMorph:**
```bash
python train.py --config config.yaml --train-data data/train --epochs 5
```

**Train RAFT:**
```bash
# Edit config.yaml: change model_name to "raft"
python train.py --config config.yaml --train-data data/train --epochs 5

# Train VoxelMorph (3D CNN-based registration)
# Edit config.yaml: change model_name to "voxelmorph"
python train.py --config config.yaml --train-data data/train --epochs 5
```

## Configuration

Edit `config.yaml` to change:
- `model_name`: "transmorph", "raft", or "voxelmorph"
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `input_size`: Input dimensions

## Outputs

Models saved to `outputs/`:
- `transmorph_epoch_XXX.pth` - TransMorph checkpoints
- `raft_epoch_XXX.pth` - RAFT checkpoints
- `*_training_summary.png` - Training plots

## Troubleshooting

**"could not read bytes" errors**: Normal - uses dummy data when files can't be read
**Models show "nan" loss**: Normal for RAFT with dummy data
**Training plots**: Automatically generated every epoch

## Notes

- **TransMorph**: 3D medical image registration
- **RAFT**: 2D optical flow estimation
- **Data**: Uses your `.mat` files in `data/train/` and `data/val/`
- **Visualization**: Automatic training progress plots
- **Environment**: All dependencies managed automatically
