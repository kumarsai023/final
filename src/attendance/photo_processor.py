import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class PhotoProcessor:
    def __init__(self):
        """Initialize the photo processor"""
        # Initialize InsightFace
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="models",
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create results directories
        self.results_dir = os.path.join("src", "attendance", "results")
        self.processed_dir = os.path.join(self.results_dir, "processed_photos")
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def process_photo(self, photo_path: str) -> Tuple[List[Dict], str]:
        """
        Process attendance photo and extract face embeddings
        Returns: List of face data and path to processed image
        """
        # Load image
        image = cv2.imread(photo_path)
        if image is None:
            raise ValueError(f"Could not load image: {photo_path}")
        
        # Get faces
        faces = self.app.get(image)
        face_data = []
        
        # Process each face
        for idx, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Store face data
            face_data.append({
                'id': idx,
                'bbox': bbox.tolist(),
                'embedding': embedding,
                'confidence': face.det_score
            })
            
            # Draw initial box (gray for unprocessed)
            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (128, 128, 128),  # Gray
                2
            )
            
            # Add face number
            cv2.putText(
                image,
                f"Face {idx+1}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                2
            )
        
        # Save intermediate result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detected_faces_{timestamp}.jpg"
        output_path = os.path.join(self.processed_dir, filename)
        cv2.imwrite(output_path, image)
        
        return face_data, output_path
    
    def mark_attendance_results(
        self,
        image_path: str,
        face_data: List[Dict],
        attendance_results: List[Dict]
    ) -> str:
        """
        Mark attendance results on the image
        Returns: Path to final processed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Process each face with attendance results
        for face, result in zip(face_data, attendance_results):
            bbox = face['bbox']
            
            # Set color based on recognition
            if result['recognized']:
                color = (0, 255, 0)  # Green for recognized
                text = f"Student {result['student_id']}"
            else:
                color = (0, 0, 255)  # Red for unknown
                text = "Unknown"
            
            # Draw box
            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )
            
            # Add text
            cv2.putText(
                image,
                text,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Save final result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_marked_{timestamp}.jpg"
        output_path = os.path.join(self.processed_dir, filename)
        cv2.imwrite(output_path, image)
        
        return output_path

def main():
    """Test the photo processor"""
    processor = PhotoProcessor()
    
    # Test photo path
    photo_path = input("\nEnter attendance photo path: ").strip()
    
    if not os.path.exists(photo_path):
        print("Error: Photo not found!")
        return
    
    try:
        # Process photo
        face_data, processed_path = processor.process_photo(photo_path)
        print(f"\nDetected {len(face_data)} faces")
        print(f"Intermediate result saved: {processed_path}")
        
        # For testing, mark all as unknown
        test_results = [
            {'recognized': False, 'student_id': None}
            for _ in face_data
        ]
        
        # Mark results
        final_path = processor.mark_attendance_results(
            processed_path,
            face_data,
            test_results
        )
        print(f"Final result saved: {final_path}")
        
    except Exception as e:
        print(f"Error processing photo: {str(e)}")

if __name__ == "__main__":
    main() 