import yaml
from pathlib import Path

class ConfigManager:
    @staticmethod
    def load_config(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Resolve paths
        base_path = Path(__file__).parent.parent
        if 'paths' in config:
            for key in config['paths']:
                config['paths'][key] = str(base_path / config['paths'][key])
                
        return config
    
    @staticmethod
    def load_and_merge_configs(config_paths):
        """Load and merge multiple configuration files"""
        merged_config = {}
        for config_path in config_paths:
            config = ConfigManager.load_config(config_path)
            merged_config.update(config)
        return merged_config

    @staticmethod
    def save_config(config, config_path):
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
