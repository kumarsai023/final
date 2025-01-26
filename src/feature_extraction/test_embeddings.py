import cv2
import numpy as np
import os
from face_processor import FaceProcessor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class EmbeddingTester:
    def __init__(self):
        """Initialize the embedding tester"""
        self.face_processor = FaceProcessor()
        self.stored_embedding = None
        
    def load_stored_embedding(self, embedding_path: str):
        """Load the stored embedding"""
        try:
            self.stored_embedding = np.load(embedding_path)
            print(f"Loaded embedding shape: {self.stored_embedding.shape}")
            return True
        except Exception as e:
            print(f"Error loading embedding: {str(e)}")
            return False
    
    def process_test_image(self, image_path: str) -> tuple:
        """Process a test image and return similarity score and face bbox"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return 0.0, None
            
            # Get face embedding
            faces = self.face_processor.app.get(img)
            
            if not faces:
                print("No face detected in test image")
                return 0.0, None
            
            # Get the largest face
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            test_embedding = face.embedding
            
            # Calculate similarity
            similarity = cosine_similarity(
                self.stored_embedding.reshape(1, -1),
                test_embedding.reshape(1, -1)
            )[0][0]
            
            return similarity, face.bbox
            
        except Exception as e:
            print(f"Error processing test image: {str(e)}")
            return 0.0, None
    
    def save_result_image(self, image_path: str, similarity: float, bbox):
        """Save the test image with similarity score and face box"""
        img = cv2.imread(image_path)
        if img is None:
            return
        
        if bbox is not None:
            # Draw face bounding box
            x1, y1, x2, y2 = [int(b) for b in bbox]
            color = (0, 255, 0) if similarity > 0.5 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add similarity score to image
        text = f"Similarity: {similarity:.2f}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0) if similarity > 0.5 else (0, 0, 255), 2)
        
        # Create results directory if it doesn't exist
        results_dir = "data/test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save image
        output_path = os.path.join(results_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")

def main():
    """Main function to test embeddings"""
    tester = EmbeddingTester()
    
    # Load stored embedding
    embedding_path = "data/embeddings/student_embedding.npy"
    if not tester.load_stored_embedding(embedding_path):
        return
    
    print("\nEmbedding Testing")
    print("================")
    
    # Test with images
    test_dir = "data/test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"\nPlease add test images to: {os.path.abspath(test_dir)}")
        return
    
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print("\nNo test images found. Please add some images to test_images directory.")
        return
    
    print("\nTesting with images...")
    for img_file in test_images:
        img_path = os.path.join(test_dir, img_file)
        print(f"\nTesting: {img_file}")
        
        # Get similarity and face bbox
        similarity, bbox = tester.process_test_image(img_path)
        print(f"Similarity score: {similarity:.4f}")
        
        # Save result image with face box
        tester.save_result_image(img_path, similarity, bbox)

if __name__ == "__main__":
    main() 