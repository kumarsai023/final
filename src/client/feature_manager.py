import numpy as np
import os
from typing import Optional, List
import logging
from datetime import datetime

class FeatureManager:
    """Manage local features for federated learning"""
    
    def __init__(self, feature_path: str = None):
        """Initialize feature manager"""
        # Use absolute path
        if feature_path is None:
            # Default to project root/data/embeddings
            current_dir = os.path.dirname(os.path.abspath(__file__))
            feature_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 
                                      'data', 'embeddings')
        
        self.feature_path = feature_path
        self.current_embedding = None
        self.embedding_timestamp = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.feature_path, exist_ok=True)
        
        self.logger.info(f"Using embedding path: {os.path.abspath(self.feature_path)}")
        
    def load_embedding(self) -> bool:
        """Load local embedding"""
        try:
            embedding_file = os.path.join(self.feature_path, "student_embedding.npy")
            if not os.path.exists(embedding_file):
                self.logger.error(f"Embedding file not found: {embedding_file}")
                return False
            
            self.current_embedding = np.load(embedding_file)
            self.embedding_timestamp = datetime.now()
            
            self.logger.info("Embedding loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading embedding: {str(e)}")
            return False
    
    def get_features(self) -> Optional[np.ndarray]:
        """Get current features for sharing"""
        if self.current_embedding is None:
            if not self.load_embedding():
                return None
        
        # Normalize embedding before sharing
        normalized = self.current_embedding / np.linalg.norm(self.current_embedding)
        return normalized
    
    def update_embedding(self, new_embedding: np.ndarray) -> bool:
        """Update local embedding"""
        try:
            self.current_embedding = new_embedding
            self.embedding_timestamp = datetime.now()
            
            # Save updated embedding
            save_path = os.path.join(self.feature_path, "student_embedding.npy")
            np.save(save_path, new_embedding)
            
            self.logger.info("Embedding updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating embedding: {str(e)}")
            return False

def main():
    """Test feature manager"""
    # Use direct path from project root
    feature_path = os.path.join("data", "embeddings")
    manager = FeatureManager(feature_path)
    
    print(f"\nChecking embedding path: {os.path.abspath(manager.feature_path)}")
    
    # Test loading
    if manager.load_embedding():
        features = manager.get_features()
        if features is not None:
            print(f"Features loaded successfully!")
            print(f"Shape: {features.shape}")
            print(f"Norm: {np.linalg.norm(features)}")
    else:
        print("\nNo embedding file found. Please run face_processor.py first to generate embeddings.")

if __name__ == "__main__":
    main() 