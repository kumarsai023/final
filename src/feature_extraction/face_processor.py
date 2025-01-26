import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

class FaceProcessor:
    def __init__(self):
        """Initialize Face Processor with InsightFace"""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(project_root, "models"),  # Updated path
            providers=['CPUExecutionProvider']
        )
        # Adjust detection size and parameters
        self.app.prepare(ctx_id=0, det_size=(320, 320))  # Reduced size
        
    def process_student_images(self, student_number: str) -> Optional[np.ndarray]:
        """Process student images and generate embedding"""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define student-specific directory using project root
        image_dir = os.path.join(project_root, "data", "student_data", f"student_{student_number}")
        embeddings = []
        
        # Get all images
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        print(f"\nProcessing {len(image_files)} images for Student {student_number}...")
        
        # Debug: Print full path
        print(f"Image directory: {os.path.abspath(image_dir)}")
        
        for img_file in image_files:
            try:
                # Load image
                img_path = os.path.join(image_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue
                
                # Debug: Print image shape
                print(f"Processing {img_file}, shape: {img.shape}")
                
                # Resize image if too large
                max_size = 1024
                if max(img.shape) > max_size:
                    scale = max_size / max(img.shape)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                
                # Get face embedding
                faces = self.app.get(img)
                
                if not faces:
                    # Try with resized image
                    resized_img = cv2.resize(img, (640, 640))
                    faces = self.app.get(resized_img)
                
                if not faces:
                    print(f"No face detected in {img_file}")
                    # Save debug image
                    debug_dir = "debug_images"
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f"failed_{img_file}"), img)
                    continue
                
                # Get the largest face
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embeddings.append(face.embedding)
                print(f"Successfully processed: {img_file}")
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        if not embeddings:
            print("No valid embeddings generated")
            return None
        
        # Calculate average embedding
        average_embedding = np.mean(embeddings, axis=0)
        # Normalize embedding
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        # Save embedding in project root's data/embeddings folder
        embedding_dir = os.path.join(project_root, "data", "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        embedding_path = os.path.join(embedding_dir, f"student_{student_number}_embedding.npy")
        np.save(embedding_path, average_embedding)
        print(f"\nEmbedding saved as: {embedding_path}")
        
        return average_embedding
    
    def process_classroom_photo(self, 
                              image_path: str,
                              min_face_size: int = 50) -> List[np.ndarray]:
        """
        Process classroom photo and extract all face embeddings
        
        Args:
            image_path: Path to classroom photo
            min_face_size: Minimum face size to consider
            
        Returns:
            List of face embeddings
        """
        embeddings = []
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not load classroom image: {image_path}")
                return embeddings
            
            # Detect faces
            faces = self.app.get(img)
            
            self.logger.info(f"Detected {len(faces)} faces in classroom photo")
            
            # Process each face
            for face in faces:
                face_width = face.bbox[2] - face.bbox[0]
                if face_width >= min_face_size:
                    embeddings.append(face.embedding)
                else:
                    self.logger.warning("Skipped a face due to small size")
            
        except Exception as e:
            self.logger.error(f"Error processing classroom photo: {str(e)}")
        
        return embeddings

def main():
    """Test the face processor"""
    print("\nFace Feature Extraction")
    print("=====================")
    
    # Get student number
    while True:
        student_number = input("\nEnter student number (e.g., 01, 02, etc.): ").strip()
        if student_number.isdigit():
            if len(student_number) == 1:
                student_number = f"0{student_number}"
            if len(student_number) == 2:
                break
        print("Please enter a valid student number (1-99)!")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create necessary directories using project root
    student_data_dir = os.path.join(project_root, "data", "student_data", f"student_{student_number}")
    if not os.path.exists(student_data_dir):
        print(f"\nError: No data found for student {student_number}")
        print(f"Please ensure images are collected in: {student_data_dir}")
        print("\nFollow these steps:")
        print("1. First collect face data using:")
        print("   python src/data_collection/face_collector.py")
        print("2. Then run feature extraction")
        return
    
    processor = FaceProcessor()
    
    print(f"\nProcessing images for Student {student_number}")
    print(f"Using directory: {student_data_dir}")
    
    embedding = processor.process_student_images(student_number)
    
    if embedding is not None:
        print("\nFeature extraction successful!")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Saved as: student_{student_number}_embedding.npy")
    else:
        print("\nFeature extraction failed!")

if __name__ == "__main__":
    main() 