import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

class EmbeddingHandler:
    def __init__(self, base_dir: str = "../../data/embeddings"):
        """
        Initialize Embedding Handler
        
        Args:
            base_dir: Base directory for storing embeddings
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_student_embedding(self, 
                             student_id: str,
                             embedding: np.ndarray,
                             metadata: Optional[Dict] = None) -> bool:
        """
        Save student's face embedding
        
        Args:
            student_id: Unique student identifier
            embedding: Face embedding array
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Create student directory
            student_dir = os.path.join(self.base_dir, f"student_{student_id}")
            os.makedirs(student_dir, exist_ok=True)
            
            # Save embedding
            embedding_path = os.path.join(student_dir, "embedding.npy")
            np.save(embedding_path, embedding)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "created_at": datetime.now().isoformat(),
                "embedding_shape": embedding.shape,
                "embedding_path": embedding_path
            })
            
            metadata_path = os.path.join(student_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"Saved embedding for student {student_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving embedding: {str(e)}")
            return False
    
    def load_student_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """
        Load student's face embedding
        
        Args:
            student_id: Student identifier
            
        Returns:
            Face embedding if exists, None otherwise
        """
        try:
            embedding_path = os.path.join(
                self.base_dir,
                f"student_{student_id}",
                "embedding.npy"
            )
            
            if os.path.exists(embedding_path):
                return np.load(embedding_path)
            
            self.logger.warning(f"No embedding found for student {student_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading embedding: {str(e)}")
            return None

def main():
    """Test the embedding handler"""
    handler = EmbeddingHandler()
    
    # Test saving
    test_embedding = np.random.rand(512)  # InsightFace uses 512-dim embeddings
    success = handler.save_student_embedding(
        "test_student",
        test_embedding,
        {"test": True}
    )
    
    if success:
        print("Test embedding saved successfully!")
        
        # Test loading
        loaded_embedding = handler.load_student_embedding("test_student")
        if loaded_embedding is not None:
            print("Test embedding loaded successfully!")
            print(f"Shape: {loaded_embedding.shape}")

if __name__ == "__main__":
    main() 