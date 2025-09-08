# Deep Learning Model Training Framework

**Simple, unified framework for training TransMorph, RAFT, and VoxelMorph models with .mat file inputs.**

## Quick Start

1. **Setup**: `./setup.sh`
2. **Activate**: `./activate.sh` 
3. **Train**: `python train.py --config config.yaml --train-data data/train --epochs 5`

## What's Included

- **TransMorph**: 3D medical image registration (Transformer-based)
- **RAFT**: 2D optical flow estimation (CNN-based)
- **VoxelMorph**: 3D medical image registration (CNN-based)
- **Automatic visualization**: Training progress plots
- **Easy configuration**: YAML-based settings
- **Data handling**: Automatic .mat file processing

## Structure

```
├── train.py          # Main training script
├── config.yaml       # Configuration
├── data/             # Your .mat files
├── outputs/          # Model checkpoints & plots
├── requirements.txt  # Dependencies
├── setup.sh          # Environment setup
└── activate.sh       # Environment activation
```

## Configuration

Edit `config.yaml`:
```yaml
model_name: transmorph  # or "raft"
num_epochs: 5
batch_size: 1
learning_rate: 0.0001
input_size: [32, 32, 32]  # TransMorph: [H,W,D], RAFT: [H,W]
```

## Usage Examples

**Train TransMorph:**
```bash
python train.py --config config.yaml --train-data data/train --epochs 10
```

**Train RAFT:**
```bash
# Change model_name to "raft" in config.yaml first
python train.py --config config.yaml --train-data data/train --epochs 10
```

## Outputs

- **Models**: `outputs/modelname_epoch_XXX.pth`
- **Plots**: `outputs/modelname_training_summary.png`
- **Progress**: Real-time training visualization

## Requirements

- Python 3.8+
- PyTorch
- scipy, numpy, matplotlib
- All dependencies in `requirements.txt`

## Notes

- Handles `.mat` files automatically
- Creates training visualizations
- Saves checkpoints every epoch
- Works with CPU (no GPU required)

**See `QUICK_START.md` for detailed instructions.**