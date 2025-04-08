import argparse
import yaml
import torch
from pathlib import Path
from data_processing.rtl_processor import RTLProcessor
from data_processing.layout_processor import LayoutProcessor
from ai_models.gnn_model import HardwareTrojanGNN
from utils.config_manager import ConfigManager
from utils.logger import TrojanLogger

def main():
    parser = argparse.ArgumentParser(description='Hardware Trojan Detection System')
    parser.add_argument('--rtl', type=str, help='Path to RTL file')
    parser.add_argument('--gdsii', type=str, help='Path to GDSII file')
    parser.add_argument('--config', type=str, default='configs/paths.yaml')
    parser.add_argument('--model-config', type=str, default='configs/model_params.yaml')
    args = parser.parse_args()

    # Load and merge configurations
    paths_config = ConfigManager.load_config(args.config)
    model_config = ConfigManager.load_config(args.model_config)
    
    # Setup logger
    logger = TrojanLogger().get_logger()
    logger.info("Starting Hardware Trojan Detection")
    
    if args.rtl:
        logger.info(f"Processing RTL file: {args.rtl}")
        rtl_processor = RTLProcessor(paths_config['rtl_params'])
        design_graph = rtl_processor.parse_verilog(args.rtl)
        embeddings = rtl_processor.generate_embeddings()
        
        # Load trained GNN model
        gnn_model = HardwareTrojanGNN(
            input_dim=model_config['gnn']['input_dim'],
            hidden_dim=model_config['gnn']['hidden_dim'],
            output_dim=model_config['gnn']['output_dim']
        )
        gnn_model.load_state_dict(torch.load(paths_config['paths']['model_weights']))
        
        # Perform inference
        # (Implementation for converting graph to PyG Data omitted for brevity)
        logger.info("RTL analysis completed")
    
    if args.gdsii:
        logger.info(f"Processing GDSII file: {args.gdsii}")
        layout_processor = LayoutProcessor(paths_config['layout_params'])
        layout_processor.load_gdsii(args.gdsii)
        
        # Detect anomalies in all layers
        all_anomalies = []
        for layer_id in paths_config['layout_params'].get('layers_to_analyze', [1]):
            anomalies = layout_processor.detect_anomalies(layer_id)
            all_anomalies.extend(anomalies)
        
        logger.info(f"Detected {len(all_anomalies)} layout anomalies:")
        for anomaly in all_anomalies:
            logger.info(f"- Class {anomaly['class']}: Confidence {anomaly['confidence']:.2f}, Bounding box {anomaly['bbox']}")
        
        logger.info("GDSII analysis completed")

if __name__ == "__main__":
    main()
