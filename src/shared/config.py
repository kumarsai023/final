import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ServerConfig:
    """Server configuration settings"""
    host: str = "localhost"
    port: int = 5000
    max_clients: int = 100
    feature_dim: int = 512  # InsightFace embedding dimension
    aggregation_threshold: int = 5  # Minimum clients for aggregation
    model_save_dir: str = "../../models/federated"

@dataclass
class ClientConfig:
    """Client configuration settings"""
    server_host: str = "localhost"
    server_port: int = 5000
    feature_path: str = "../../data/embeddings"
    client_id: str = ""
    retry_limit: int = 3
    retry_delay: int = 5  # seconds

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    key_size: int = 2048
    key_dir: str = "../../keys"
    cert_dir: str = "../../certificates"
    protocol: str = "TLS"

class FederatedConfig:
    """Main configuration class"""
    def __init__(self):
        self.server = ServerConfig()
        self.client = ClientConfig()
        self.security = SecurityConfig()
        
        # Create necessary directories
        os.makedirs(self.server.model_save_dir, exist_ok=True)
        os.makedirs(self.security.key_dir, exist_ok=True)
        os.makedirs(self.security.cert_dir, exist_ok=True)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def load_config(cls, config_path: str = None):
        """Load configuration from file"""
        config = cls()
        if config_path and os.path.exists(config_path):
            # Load from file if needed
            pass
        return config

def get_config() -> FederatedConfig:
    """Get configuration instance"""
    return FederatedConfig() 