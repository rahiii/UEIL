#!/bin/bash

# UEIL_DL Environment Activation Script
echo "🔧 Activating UEIL_DL environment..."

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate
echo "✅ Environment activated!"
echo "🐍 Python: $(which python)"
echo "📦 Pip: $(which pip)"
echo ""
echo "To deactivate: deactivate"
echo "To start training: python train.py --config config.yaml --train-data data/train"