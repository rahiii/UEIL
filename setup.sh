#!/bin/bash

# UEIL_DL Environment Setup Script
echo "Setting up UEIL_DL environment..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python 3.8+ required. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Clone repositories if they don't exist
if [ ! -d "TransMorph_repo" ]; then
    echo "Cloning TransMorph repository..."
    git clone https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git TransMorph_repo
fi

if [ ! -d "RAFT_repo" ]; then
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git RAFT_repo
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/train/fixed data/train/moving data/val/fixed data/val/moving
mkdir -p checkpoints outputs

echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To start training:"
echo "  python train.py --config config.yaml --train-data data/train"