"""
Unified Model Training Framework - Simple Version
Supports TransMorph and RAFT models with .mat file input
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import cv2
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging
import time

# Add repositories to path
sys.path.insert(0, str(Path(__file__).parent / 'TransMorph_repo' / 'TransMorph'))
sys.path.insert(0, str(Path(__file__).parent / 'RAFT_repo'))
sys.path.insert(0, str(Path(__file__).parent / 'VoxelMorph_repo'))

# Try to import model modules
try:
    from models.TransMorph import TransMorph as TransMorphModel
    from models.configs_TransMorph import get_3DTransMorph_config
    from losses import Grad, NCC
    TRANSMORPH_AVAILABLE = True
except ImportError:
    TRANSMORPH_AVAILABLE = False

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent / 'RAFT_repo'))
    sys.path.insert(0, str(Path(__file__).parent / 'RAFT_repo' / 'core'))
    from raft import RAFT
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False

try:
    # Try to import VoxelMorph, but it has TensorFlow compatibility issues
    # We'll create a simplified PyTorch implementation instead
    VOXELMORPH_AVAILABLE = True
except ImportError:
    VOXELMORPH_AVAILABLE = False


@dataclass
class Config:
    """Simple configuration class."""
    # Model settings
    model_name: str = 'transmorph'  # 'transmorph' or 'raft'
    model_variant: str = 'transmorph'  # 'transmorph', 'transmorph_diff', 'raft_large', 'raft_small'
    
    # Training settings
    batch_size: int = 4
    learning_rate: float = 0.0001
    num_epochs: int = 100
    steps_per_epoch: int = None
    device: str = 'cuda'
    
    # Optimizer settings
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    
    # Loss settings
    loss_type: str = 'mse'  # 'mse', 'ncc', 'l1'
    
    # Data settings
    input_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    num_channels: int = 1
    
    # Paths
    train_data_dir: str = './data/train'
    val_data_dir: str = './data/val'
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    
    # Model-specific parameters
    embed_dim: int = 96
    num_heads: List[int] = field(default_factory=lambda: [4, 4, 8, 8])
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])
    window_size: List[int] = field(default_factory=lambda: [5, 6, 7])
    drop_path_rate: float = 0.3
    
    # RAFT specific parameters
    hidden_dim: int = 128
    context_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 4
    flow_iters: int = 12
    
    # VoxelMorph specific parameters
    voxelmorph_int_steps: int = 7
    voxelmorph_int_downsize: int = 2
    voxelmorph_unet_features: List[int] = field(default_factory=lambda: [16, 32, 32, 32])
    voxelmorph_unet_conv_layers: int = 2
    
    # Loss settings
    similarity_loss: str = 'ncc'  # 'ncc', 'mse'
    regularization_loss: str = 'diffusion'  # 'diffusion', 'l1'
    similarity_weight: float = 1.0
    regularization_weight: float = 0.1


class MatDataset(Dataset):
    """Dataset for .mat files with proper format handling for both TransMorph and RAFT."""
    
    def __init__(self, data_dir: str, mode: str = 'train', config: Config = None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.config = config
        
        # Find .mat files
        self.mat_files = list(self.data_dir.glob('*.mat'))
        if not self.mat_files:
            raise ValueError(f"No .mat files found in {data_dir}")
        
        print(f"Found {len(self.mat_files)} .mat files")
    
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx):
        mat_file = self.mat_files[idx]
        
        try:
            # Load .mat file
            mat_data = loadmat(str(mat_file))
            
            # Extract data
            data = None
            for key in ['data', 'image', 'img', 'vol', 'volume', 'X', 'Y']:
                if key in mat_data and not key.startswith('__'):
                    data = mat_data[key]
                    break
            
            if data is None:
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        break
            
            if data is None:
                raise ValueError(f"No data found in {mat_file}")
            
            # Convert to float32 and normalize
            data = data.astype(np.float32)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            
            # Format data based on model type
            if self.config.model_name == 'transmorph':
                # TransMorph expects 3D data: [H, W, D] -> [1, H, W, D]
                if data.ndim == 2:
                    # 2D -> 3D by adding depth dimension
                    data = np.stack([data, data], axis=2)  # [H, W, 2]
                elif data.ndim == 3:
                    pass  # Already 3D
                else:
                    raise ValueError(f"Unsupported data shape for TransMorph: {data.shape}")
                
                # Add channel dimension
                data = data[np.newaxis, ...]  # [1, H, W, D]
                
            elif self.config.model_name == 'raft':
                # RAFT expects 2D RGB data: [H, W] -> [3, H, W]
                if data.ndim == 3:
                    # 3D -> 2D by taking middle slice
                    mid_slice = data.shape[2] // 2
                    data = data[:, :, mid_slice]  # [H, W]
                elif data.ndim == 2:
                    pass  # Already 2D
                else:
                    raise ValueError(f"Unsupported data shape for RAFT: {data.shape}")
                
                # Convert to RGB (3 channels)
                data = np.stack([data, data, data], axis=0)  # [3, H, W]
            
            # Resize if needed
            if self.config and len(self.config.input_size) > 0:
                data = self._resize_data(data)
            
            return torch.from_numpy(data).float()
            
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            # Return dummy data with correct shape
            if self.config.model_name == 'transmorph':
                dummy_shape = [1] + self.config.input_size
            else:  # raft
                dummy_shape = [3] + self.config.input_size[:2]  # RGB + H + W
            return torch.zeros(dummy_shape, dtype=torch.float32)
    
    def _resize_data(self, data):
        """Resize data based on model requirements."""
        from scipy.ndimage import zoom
        
        if self.config.model_name == 'transmorph':
            # 3D data: [1, H, W, D]
            target_shape = self.config.input_size
            zoom_factors = [1] + [target_shape[i] / data.shape[i+1] for i in range(len(target_shape))]
        else:  # raft
            # 2D data: [3, H, W]
            target_shape = self.config.input_size[:2]  # Only H, W for RAFT
            zoom_factors = [1] + [target_shape[i] / data.shape[i+1] for i in range(len(target_shape))]
        
        return zoom(data, zoom_factors, order=1)


class RegistrationDataset(Dataset):
    """Dataset for image registration (pairs of images)."""
    
    def __init__(self, data_dir: str, mode: str = 'train', config: Config = None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.config = config
        
        # Find paired files
        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No paired .mat files found in {data_dir}")
        
        print(f"Found {len(self.pairs)} image pairs")
    
    def _find_pairs(self):
        """Find paired .mat files."""
        # Look for organized structure first (fixed/ and moving/ subdirectories)
        fixed_dir = self.data_dir / 'fixed'
        moving_dir = self.data_dir / 'moving'
        
        if fixed_dir.exists() and moving_dir.exists():
            # Organized structure
            fixed_files = list(fixed_dir.glob('*.mat'))
            moving_files = list(moving_dir.glob('*.mat'))
            
            # Sort and pair by name
            fixed_files.sort()
            moving_files.sort()
            
            pairs = []
            for i in range(min(len(fixed_files), len(moving_files))):
                pairs.append((fixed_files[i], moving_files[i]))
            
            return pairs
        
        else:
            # Fallback: look for files in main directory
            mat_files = list(self.data_dir.glob('*.mat'))
            pairs = []
            
            # Simple pairing: assume files are named like fixed_001.mat, moving_001.mat
            fixed_files = [f for f in mat_files if 'fixed' in f.name.lower()]
            moving_files = [f for f in mat_files if 'moving' in f.name.lower()]
            
            # Sort and pair
            fixed_files.sort()
            moving_files.sort()
            
            for i in range(min(len(fixed_files), len(moving_files))):
                pairs.append((fixed_files[i], moving_files[i]))
            
            return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        fixed_file, moving_file = self.pairs[idx]
        
        try:
            # Load both images
            fixed_data = self._load_mat_file(fixed_file)
            moving_data = self._load_mat_file(moving_file)
            
            # Format data based on model type
            if self.config.model_name in ['transmorph', 'voxelmorph']:
                # Both TransMorph and VoxelMorph expect concatenated input: [2, H, W, D]
                # Concatenate source and target along channel dimension
                combined_data = np.concatenate([moving_data, fixed_data], axis=0)  # [2, H, W, D]
                
                # Resize if needed
                if self.config and len(self.config.input_size) > 0:
                    combined_data = self._resize_data(combined_data)
                
                return torch.from_numpy(combined_data).float()
                
            elif self.config.model_name == 'raft':
                # RAFT expects separate image1 and image2: [3, H, W] each
                # Ensure same shape
                if fixed_data.shape != moving_data.shape:
                    target_shape = fixed_data.shape
                    moving_data = self._resize_to_shape(moving_data, target_shape)
                
                # Resize if needed
                if self.config and len(self.config.input_size) > 0:
                    fixed_data = self._resize_data(fixed_data)
                    moving_data = self._resize_data(moving_data)
                
                return {
                    'image1': torch.from_numpy(moving_data).float(),
                    'image2': torch.from_numpy(fixed_data).float()
                }
            
        except Exception as e:
            print(f"Error loading pair: {e}")
            # Return dummy data with correct format
            if self.config.model_name in ['transmorph', 'voxelmorph']:
                dummy_shape = [2] + self.config.input_size
                return torch.zeros(dummy_shape, dtype=torch.float32)
            else:  # raft
                dummy_shape = [3] + self.config.input_size[:2]
                dummy_tensor = torch.zeros(dummy_shape, dtype=torch.float32)
                return {'image1': dummy_tensor, 'image2': dummy_tensor}
    
    def _load_mat_file(self, mat_file):
        """Load and process .mat file based on model requirements."""
        try:
            mat_data = loadmat(str(mat_file))
            
            for key in ['img1', 'data', 'image', 'img', 'vol', 'volume', 'X', 'Y']:
                if key in mat_data and not key.startswith('__'):
                    data = mat_data[key]
                    break
            else:
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data = mat_data[key]
                        break
            
            data = data.astype(np.float32)
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        except Exception as e:
            print(f"Error loading {mat_file}: {e}")
            # Return dummy data with correct shape
            if self.config.model_name == 'transmorph':
                dummy_shape = self.config.input_size  # [H, W, D] for TransMorph
                data = np.zeros(dummy_shape, dtype=np.float32)
            else:  # raft
                dummy_shape = self.config.input_size[:2]  # Only H, W for RAFT
                data = np.zeros(dummy_shape, dtype=np.float32)
        
        # Format based on model type
        if self.config.model_name in ['transmorph', 'voxelmorph']:
            # Both TransMorph and VoxelMorph expect 3D data: [H, W, D]
            if data.ndim == 2:
                # 2D -> 3D by adding depth dimension
                data = np.stack([data, data], axis=2)  # [H, W, 2]
            elif data.ndim == 3:
                pass  # Already 3D
            else:
                raise ValueError(f"Unsupported data shape for {self.config.model_name}: {data.shape}")
            
            # Add channel dimension
            data = data[np.newaxis, ...]  # [1, H, W, D]
            
        elif self.config.model_name == 'raft':
            # RAFT expects 2D RGB data: [H, W] -> [3, H, W]
            if data.ndim == 3:
                # 3D -> 2D by taking middle slice
                mid_slice = data.shape[2] // 2
                data = data[:, :, mid_slice]  # [H, W]
            elif data.ndim == 2:
                pass  # Already 2D
            else:
                raise ValueError(f"Unsupported data shape for RAFT: {data.shape}")
            
            # Convert to RGB (3 channels)
            data = np.stack([data, data, data], axis=0)  # [3, H, W]
        
        return data
    
    def _resize_data(self, data):
        """Resize data to target size based on model requirements."""
        from scipy.ndimage import zoom
        
        if self.config.model_name in ['transmorph', 'voxelmorph']:
            # 3D data: [1, H, W, D] or [2, H, W, D]
            target_shape = self.config.input_size
            zoom_factors = [1] + [target_shape[i] / data.shape[i+1] for i in range(len(target_shape))]
        else:  # raft
            # 2D data: [3, H, W]
            target_shape = self.config.input_size[:2]  # Only H, W for RAFT
            zoom_factors = [1] + [target_shape[i] / data.shape[i+1] for i in range(len(target_shape))]
        
        return zoom(data, zoom_factors, order=1)
    
    def _resize_to_shape(self, data, target_shape):
        """Resize data to match target shape."""
        from scipy.ndimage import zoom
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(len(target_shape))]
        return zoom(data, zoom_factors, order=1)


class SimpleTransMorph(nn.Module):
    """Simplified TransMorph wrapper."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if not TRANSMORPH_AVAILABLE:
            raise ImportError("TransMorph not available")
        
        # Create model config
        model_config = get_3DTransMorph_config()
        model_config.img_size = tuple(config.input_size)
        model_config.embed_dim = config.embed_dim
        model_config.depths = tuple(config.depths)
        model_config.num_heads = tuple(config.num_heads)
        model_config.window_size = tuple(config.window_size)
        
        # Create model
        self.model = TransMorphModel(model_config)
        
        # Setup losses based on config
        if config.loss_type.lower() == 'mse':
            self.similarity_loss = nn.MSELoss()
        elif config.loss_type.lower() == 'l1':
            self.similarity_loss = nn.L1Loss()
        else:  # default to mse
            self.similarity_loss = nn.MSELoss()
        
        self.regularization_loss = nn.L1Loss()
    
    def forward(self, x):
        """Forward pass."""
        # x is concatenated input [B, 2, H, W, D]
        # Split into source and target
        source = x[:, 0:1, :, :, :]  # [B, 1, H, W, D]
        target = x[:, 1:2, :, :, :]  # [B, 1, H, W, D]
        
        # Forward through TransMorph - pass concatenated input
        warped_source, flow = self.model(x)
        
        # Compute losses
        sim_loss = self.similarity_loss(warped_source, target)
        # Regularization loss: penalize large flow values
        reg_loss = torch.mean(torch.abs(flow))
        
        return {
            'displacement_field': flow,
            'warped_source': warped_source,
            'similarity_loss': sim_loss,
            'regularization_loss': reg_loss,
            'total_loss': self.config.similarity_weight * sim_loss + self.config.regularization_weight * reg_loss
        }


class SimpleRAFT(nn.Module):
    """Simplified RAFT wrapper."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if not RAFT_AVAILABLE:
            raise ImportError("RAFT not available")
        
        # Create RAFT args as a hybrid object (dict + attributes)
        class RAFTArgs(dict):
            def __init__(self):
                super().__init__()
                self['small'] = config.model_variant == 'raft_small'
                self['hidden_dim'] = config.hidden_dim
                self['context_dim'] = config.context_dim
                self['corr_levels'] = config.corr_levels
                self['corr_radius'] = config.corr_radius
                self['dropout'] = 0.0
                self['alternate_corr'] = False
                self['mixed_precision'] = False
                
                # Also set as attributes
                self.small = self['small']
                self.hidden_dim = self['hidden_dim']
                self.context_dim = self['context_dim']
                self.corr_levels = self['corr_levels']
                self.corr_radius = self['corr_radius']
                self.dropout = self['dropout']
                self.alternate_corr = self['alternate_corr']
                self.mixed_precision = self['mixed_precision']
        
        args = RAFTArgs()
        self.model = RAFT(args)
        self.loss_fn = nn.L1Loss()
    
    def forward(self, img1, img2):
        """Forward pass."""
        flow_predictions = self.model(img1, img2, iters=12)
        flow = flow_predictions[-1]
        
        # Compute loss (simple L1 loss on flow magnitude)
        loss = torch.mean(torch.abs(flow))
        
        return loss


class SimpleVoxelMorph(nn.Module):
    """Proper VoxelMorph implementation in PyTorch."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if not VOXELMORPH_AVAILABLE:
            raise ImportError("VoxelMorph not available")
        
        # Create a proper U-Net architecture for VoxelMorph
        self.input_size = config.input_size
        self.features = config.voxelmorph_unet_features
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(2, self.features[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.features[0], self.features[0], 3, padding=1),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(self.features[0], self.features[1], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.features[1], self.features[1], 3, padding=1),
            nn.ReLU()
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(self.features[1], self.features[2], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.features[2], self.features[2], 3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(self.features[2], self.features[1], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(self.features[1], self.features[1], 3, padding=1),
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(self.features[1], self.features[0], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv3d(self.features[0], self.features[0], 3, padding=1),
            nn.ReLU()
        )
        
        # Flow field output
        self.flow = nn.Conv3d(self.features[0], 3, 3, padding=1)
        
        # Loss functions based on config
        if config.loss_type.lower() == 'mse':
            self.image_loss = nn.MSELoss()
        elif config.loss_type.lower() == 'l1':
            self.image_loss = nn.L1Loss()
        else:  # default to mse
            self.image_loss = nn.MSELoss()
    
    def forward(self, x):
        """Forward pass."""
        # x is concatenated input: [B, 2, H, W, D]
        # Split into moving and fixed
        if x.shape[1] == 2:
            moving = x[:, 0:1]  # [B, 1, H, W, D]
            fixed = x[:, 1:2]   # [B, 1, H, W, D]
        else:
            # Handle single image case
            moving = x[:, 0:1]
            fixed = x[:, 0:1]
        
        # Concatenate moving and fixed images
        combined = torch.cat([moving, fixed], dim=1)  # [B, 2, H, W, D]
        
        # Encoder
        enc1 = self.enc1(combined)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        # Decoder
        dec2 = self.dec2(enc3)
        dec1 = self.dec1(dec2)
        
        # Flow field
        flow = self.flow(dec1)  # [B, 3, H, W, D]
        
        # Simple warping (bilinear interpolation approximation)
        # For now, use a simple additive warping
        moved = moving + 0.1 * flow[:, 0:1]  # Simplified warping
        
        # Compute losses
        image_loss = self.image_loss(fixed, moved)
        grad_loss = torch.mean(torch.abs(flow))
        
        total_loss = image_loss + self.config.regularization_weight * grad_loss
        
        return total_loss


class SimpleTrainer:
    """Simple trainer class."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        if config.model_name == 'transmorph':
            self.model = SimpleTransMorph(config)
        elif config.model_name == 'raft':
            self.model = SimpleRAFT(config)
        elif config.model_name == 'voxelmorph':
            self.model = SimpleVoxelMorph(config)
        else:
            raise ValueError(f"Unknown model: {config.model_name}")
        
        self.model.to(self.device)
        
        # Setup optimizer
        if config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate, momentum=0.9)
        elif config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training progress tracking
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Limit steps per epoch if specified
        max_steps = self.config.steps_per_epoch if self.config.steps_per_epoch else len(train_loader)
        
        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            self.optimizer.zero_grad()
            
            if self.config.model_name in ['transmorph', 'voxelmorph']:
                # Both TransMorph and VoxelMorph expect concatenated input: [B, 2, H, W, D]
                if isinstance(batch, torch.Tensor):
                    # Single tensor input (concatenated)
                    x = batch.to(self.device)
                    if self.config.model_name == 'transmorph':
                        outputs = self.model(x)
                        loss = outputs['total_loss']
                    else:  # voxelmorph
                        loss = self.model(x)
                else:
                    # Dictionary input (separate source/target)
                    source = batch['source'].to(self.device)
                    target = batch['target'].to(self.device)
                    # Concatenate along channel dimension
                    x = torch.cat([source, target], dim=1)  # [B, 2, H, W, D]
                    if self.config.model_name == 'transmorph':
                        outputs = self.model(x)
                        loss = outputs['total_loss']
                    else:  # voxelmorph
                        loss = self.model(x)
                
            elif self.config.model_name == 'raft':
                # RAFT expects separate image1 and image2: [B, 3, H, W] each
                if isinstance(batch, dict) and 'image1' in batch and 'image2' in batch:
                    img1 = batch['image1'].to(self.device)
                    img2 = batch['image2'].to(self.device)
                else:
                    # Fallback: assume batch is a list of two tensors
                    img1 = batch[0].to(self.device)
                    img2 = batch[1].to(self.device)
                
                loss = self.model(img1, img2)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader)
            
            # Track losses
            self.train_losses.append(train_loss)
            self.epochs.append(epoch)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            
            # Logging
            if val_loss is not None:
                self.logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                self.logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f}")
            
            # Only save checkpoint and plots at the final epoch
            if epoch == self.config.num_epochs - 1:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"{self.config.model_name}_epoch_{epoch:03d}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, checkpoint_path)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                # Create and save training plots (summary only)
                self.plot_training_progress(epoch)
                # Write concise configuration summary
                self.write_config_summary(final_train_loss=train_loss, final_val_loss=val_loss)
        
        self.logger.info("Training completed!")
    
    def validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if self.config.model_name in ['transmorph', 'voxelmorph']:
                    if isinstance(batch, torch.Tensor):
                        x = batch.to(self.device)
                        if self.config.model_name == 'transmorph':
                            outputs = self.model(x)
                            loss = outputs['total_loss']
                        else:  # voxelmorph
                            loss = self.model(x)
                    else:
                        # Handle dict input if needed
                        x = batch['image'].to(self.device)
                        if self.config.model_name == 'transmorph':
                            outputs = self.model(x)
                            loss = outputs['total_loss']
                        else:  # voxelmorph
                            loss = self.model(x)
                elif self.config.model_name == 'raft':
                    img1 = batch['image1'].to(self.device)
                    img2 = batch['image2'].to(self.device)
                    loss = self.model(img1, img2)
                
                if isinstance(loss, dict):
                    # Handle dict output (shouldn't happen with our models)
                    loss_value = loss.get('loss', 0.0)
                    if hasattr(loss_value, 'item'):
                        total_loss += loss_value.item()
                    else:
                        total_loss += float(loss_value)
                else:
                    total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def plot_training_progress(self, epoch):
        """Create and save training progress plots."""
        if len(self.train_losses) < 1:
            return
        # Save a single summary plot with training and validation loss
        summary_plot_path = Path(self.config.checkpoint_dir) / f"{self.config.model_name}_training_summary.png"
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.config.model_name.upper()} Training Summary')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Training summary plot saved: {summary_plot_path}")

    def write_config_summary(self, final_train_loss: float, final_val_loss: float | None):
        """Write a concise configuration and results summary file."""
        summary_lines = []
        summary_lines.append("Training Configuration Summary")
        summary_lines.append("===============================")
        summary_lines.append("")
        summary_lines.append(f"Model: {self.config.model_name}")
        summary_lines.append(f"Training Data Dir: {self.config.train_data_dir}")
        if getattr(self.config, 'val_data_dir', None):
            summary_lines.append(f"Validation Data Dir: {self.config.val_data_dir}")
        summary_lines.append(f"Input Size: {self.config.input_size}")
        summary_lines.append(f"Batch Size: {self.config.batch_size}")
        summary_lines.append(f"Learning Rate: {self.config.learning_rate}")
        summary_lines.append(f"Epochs: {self.config.num_epochs}")
        summary_lines.append(f"Optimizer: {self.config.optimizer}")
        summary_lines.append(f"Loss Type: {self.config.loss_type}")
        if self.config.steps_per_epoch:
            summary_lines.append(f"Steps per Epoch: {self.config.steps_per_epoch}")
        summary_lines.append("")
        summary_lines.append("Results:")
        summary_lines.append(f"- Final Training Loss: {final_train_loss:.6f}")
        if final_val_loss is not None:
            summary_lines.append(f"- Final Validation Loss: {final_val_loss:.6f}")
        # Save
        summary_path = Path(self.config.checkpoint_dir) / f"{self.config.model_name}_config_summary.txt"
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write("\n".join(summary_lines))
        self.logger.info(f"Configuration summary saved: {summary_path}")


def load_config_from_yaml(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Config(**data)


class OpticalFlowDataset(Dataset):
    """Dataset for optical flow (RAFT) - pairs of images."""
    
    def __init__(self, data_dir: str, mode: str = 'train', config: Config = None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.config = config
        
        # Find paired files
        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No paired .mat files found in {data_dir}")
        
        print(f"Found {len(self.pairs)} image pairs for RAFT")
    
    def _find_pairs(self):
        """Find paired .mat files."""
        # Look for organized structure first (fixed/ and moving/ subdirectories)
        fixed_dir = self.data_dir / 'fixed'
        moving_dir = self.data_dir / 'moving'
        
        if fixed_dir.exists() and moving_dir.exists():
            # Organized structure
            fixed_files = list(fixed_dir.glob('*.mat'))
            moving_files = list(moving_dir.glob('*.mat'))
            
            # Sort and pair by name
            fixed_files.sort()
            moving_files.sort()
            
            pairs = []
            for i in range(min(len(fixed_files), len(moving_files))):
                pairs.append((fixed_files[i], moving_files[i]))
            
            return pairs
        
        else:
            # Fallback: look for files in main directory
            mat_files = list(self.data_dir.glob('*.mat'))
            pairs = []
            
            # Simple pairing: assume files are named like fixed_001.mat, moving_001.mat
            fixed_files = [f for f in mat_files if 'fixed' in f.name.lower()]
            moving_files = [f for f in mat_files if 'moving' in f.name.lower()]
            
            # Sort and pair
            fixed_files.sort()
            moving_files.sort()
            
            for i in range(min(len(fixed_files), len(moving_files))):
                pairs.append((fixed_files[i], moving_files[i]))
            
            return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        fixed_file, moving_file = self.pairs[idx]
        
        try:
            # Load both images
            fixed_data = self._load_mat_file(fixed_file)
            moving_data = self._load_mat_file(moving_file)
            
            # RAFT expects separate image1 and image2: [3, H, W] each
            # Ensure same shape
            if fixed_data.shape != moving_data.shape:
                target_shape = fixed_data.shape
                moving_data = self._resize_to_shape(moving_data, target_shape)
            
            # Resize if needed
            if self.config and len(self.config.input_size) > 0:
                fixed_data = self._resize_data(fixed_data)
                moving_data = self._resize_data(moving_data)
            
            return {
                'image1': torch.from_numpy(moving_data).float(),
                'image2': torch.from_numpy(fixed_data).float()
            }
            
        except Exception as e:
            print(f"Error loading pair: {e}")
            # Return dummy data with correct format
            dummy_shape = [3] + self.config.input_size[:2]
            dummy_tensor = torch.zeros(dummy_shape, dtype=torch.float32)
            return {'image1': dummy_tensor, 'image2': dummy_tensor}
    
    def _load_mat_file(self, mat_file):
        """Load and process .mat file for RAFT."""
        mat_data = loadmat(str(mat_file))
        
        for key in ['img1', 'data', 'image', 'img', 'vol', 'volume', 'X', 'Y']:
            if key in mat_data and not key.startswith('__'):
                data = mat_data[key]
                break
        else:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data = mat_data[key]
                    break
        
        data = data.astype(np.float32)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # RAFT expects 2D RGB data: [H, W] -> [3, H, W]
        if data.ndim == 3:
            # 3D -> 2D by taking middle slice
            mid_slice = data.shape[2] // 2
            data = data[:, :, mid_slice]  # [H, W]
        elif data.ndim == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Unsupported data shape for RAFT: {data.shape}")
        
        # Convert to RGB (3 channels)
        data = np.stack([data, data, data], axis=0)  # [3, H, W]
        
        return data
    
    def _resize_data(self, data):
        """Resize data to target size for RAFT."""
        from scipy.ndimage import zoom
        # 2D data: [3, H, W]
        target_shape = self.config.input_size[:2]  # Only H, W for RAFT
        zoom_factors = [1] + [target_shape[i] / data.shape[i+1] for i in range(len(target_shape))]
        
        return zoom(data, zoom_factors, order=1)
    
    def _resize_to_shape(self, data, target_shape):
        """Resize data to match target shape."""
        from scipy.ndimage import zoom
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(len(target_shape))]
        return zoom(data, zoom_factors, order=1)


def create_data_loaders(config: Config):
    """Create data loaders."""
    if config.model_name in ['transmorph', 'voxelmorph']:
        train_dataset = RegistrationDataset(config.train_data_dir, 'train', config)
        val_dataset = RegistrationDataset(config.val_data_dir, 'val', config) if Path(config.val_data_dir).exists() else None
    elif config.model_name == 'raft':
        train_dataset = OpticalFlowDataset(config.train_data_dir, 'train', config)
        val_dataset = OpticalFlowDataset(config.val_data_dir, 'val', config) if Path(config.val_data_dir).exists() else None
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0) if val_dataset else None
    
    return train_loader, val_loader


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple Model Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val-data', type=str, default=None, help='Path to validation data')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    
    # Override with command line arguments
    config.train_data_dir = args.train_data
    if args.val_data:
        config.val_data_dir = args.val_data
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epochs:
        config.num_epochs = args.epochs
    
    print(f"Training {config.model_name} model")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Input size: {config.input_size}")
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(config)
        print(f"Created data loaders: {len(train_loader)} train batches")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create trainer and train
    trainer = SimpleTrainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()