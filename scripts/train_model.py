import argparse
import yaml
import torch
import os
import glob
from pathlib import Path
from ai_models.gnn_model import HardwareTrojanGNN, GNNTrainer
from data_processing.rtl_processor import RTLProcessor
from torch_geometric.data import DataLoader, Dataset
from utils.config_manager import ConfigManager
from utils.logger import TrojanLogger

class RTLGraphDataset(Dataset):
    def __init__(self, root_dir, rtl_params, transform=None):
        self.root_dir = root_dir
        self.rtl_params = rtl_params
        self.transform = transform
        self.file_paths = glob.glob(os.path.join(root_dir, "**/*.v"), recursive=True)
        self.processor = RTLProcessor(rtl_params)
        self.labels = self._load_labels()
        
    def _load_labels(self):
        """Load labels from label file or generate based on filename"""
        label_file = os.path.join(self.root_dir, "labels.txt")
        labels = {}
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        filename, label = parts
                        labels[filename] = int(label)
        else:
            # Assume files with "trojan" in name are positive examples
            for file_path in self.file_paths:
                filename = os.path.basename(file_path)
                labels[filename] = 1 if "trojan" in filename.lower() else 0
                
        return labels
    
    def len(self):
        return len(self.file_paths)
    
    def get(self, idx):
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        # Process RTL file
        graph = self.processor.parse_verilog(file_path)
        embeddings = self.processor.generate_embeddings()
        
        # Convert to PyG data format
        # This is a simplified version - actual implementation would depend on graph structure
        edge_index = torch.tensor([[s, t] for s, t in graph.edges()]).t().contiguous()
        x = torch.tensor([embeddings[node] for node in graph.nodes()])
        y = torch.tensor([self.labels.get(filename, 0)])
        
        return {'x': x, 'edge_index': edge_index, 'y': y}

def load_datasets(data_path, rtl_params, batch_size=32, val_split=0.2):
    """Load RTL datasets for training and validation"""
    dataset = RTLGraphDataset(data_path, rtl_params)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_gnn(config_path, paths_config_path):
    # Load configurations
    model_config = ConfigManager.load_config(config_path)
    paths_config = ConfigManager.load_config(paths_config_path)
    
    # Setup logger
    logger = TrojanLogger().get_logger()
    logger.info("Starting GNN model training")
    
    # Initialize components
    model = HardwareTrojanGNN(
        input_dim=model_config['gnn']['input_dim'],
        hidden_dim=model_config['gnn']['hidden_dim'],
        output_dim=model_config['gnn']['output_dim']
    )
    trainer = GNNTrainer(model)
    
    # Load dataset
    train_loader, val_loader = load_datasets(
        paths_config['paths']['rtl_samples'], 
        paths_config['rtl_params'],
        batch_size=model_config['gnn'].get('batch_size', 32)
    )
    
    # Create output directory for model weights
    weights_dir = Path(paths_config['paths']['model_weights']).parent
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(model_config['gnn'].get('epochs', 100)):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), paths_config['paths']['model_weights'])
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    logger.info("Training completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model_params.yaml')
    parser.add_argument('--paths-config', default='configs/paths.yaml')
    args = parser.parse_args()
    train_gnn(args.config, args.paths_config)
