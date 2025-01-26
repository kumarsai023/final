import cv2
import numpy as np
import os
import sys
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
from typing import Tuple, List, Dict

class BatchFaceRecognitionTester:
    def __init__(self):
        """Initialize batch face recognition tester"""
        # Initialize InsightFace
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="models",
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load global model
        self.load_latest_global_model()
        
        # Colors for visualization
        self.GREEN = (0, 255, 0)  # BGR format
        self.RED = (0, 0, 255)    # BGR format
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("src", "test", "results", timestamp)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("\nBatch Face Recognition Tester")
        print("===========================")
    
    def load_latest_global_model(self):
        """Load the latest global model"""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        global_model_dir = os.path.join(project_root, "data", "global_model")
        
        model_files = [f for f in os.listdir(global_model_dir) 
                      if f.startswith('global_model_')]
        
        if not model_files:
            raise Exception("No global model found!")
        
        latest_model = os.path.join(global_model_dir, sorted(model_files)[-1])
        print(f"\nLoading global model: {latest_model}")
        
        # Load the model data
        self.model_data = np.load(latest_model, allow_pickle=True)
        self.student_embeddings = {}
        
        # Load student embeddings
        student_ids = self.model_data['student_ids']
        for student_id in student_ids:
            self.student_embeddings[str(student_id)] = self.model_data[str(student_id)]
        
        print(f"Loaded embeddings for {len(self.student_embeddings)} students")
    
    def check_face(self, face_embedding: np.ndarray, threshold: float = 0.7) -> Tuple[bool, str, float]:
        """Check a face against stored embeddings"""
        best_match = {'student_id': None, 'similarity': 0.0}
        
        for student_id, stored_embedding in self.student_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(face_embedding, stored_embedding) / \
                        (np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding))
            
            if similarity > best_match['similarity']:
                best_match = {
                    'student_id': student_id,
                    'similarity': similarity
                }
        
        is_match = best_match['similarity'] >= threshold
        return is_match, best_match['student_id'], best_match['similarity']
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single image and return results"""
        results = []
        
        # Get faces
        faces = self.app.get(image)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Check against global model
            is_match, student_id, confidence = self.check_face(embedding)
            
            # Determine color and text based on match
            if is_match:
                color = self.GREEN
                text = f"Student {student_id}"
            else:
                color = self.RED
                text = "Unknown"
            
            # Draw rectangle
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
            
            # Store result
            results.append({
                'bbox': bbox.tolist(),
                'student_id': student_id if is_match else None,
                'confidence': confidence,
                'recognized': is_match
            })
        
        return image, results
    
    def process_test_folder(self, test_folder: str):
        """Process all images in test folder"""
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png')
        
        # Results summary
        summary = []
        
        # Process each image
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(image_extensions):
                print(f"\nProcessing: {filename}")
                
                # Load image
                image_path = os.path.join(test_folder, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Error loading image: {filename}")
                    continue
                
                # Process image
                processed_image, results = self.process_image(image)
                
                # Save processed image
                output_path = os.path.join(self.results_dir, f"processed_{filename}")
                cv2.imwrite(output_path, processed_image)
                
                # Add to summary
                summary.append({
                    'filename': filename,
                    'results': results
                })
                
                print(f"Results saved: processed_{filename}")
                for result in results:
                    if result['recognized']:
                        print(f"- Student {result['student_id']} (Confidence: {result['confidence']:.2f})")
                    else:
                        print("- Unknown person")
        
        # Save summary
        self.save_summary(summary)
        
        print(f"\nAll results saved in: {self.results_dir}")
    
    def save_summary(self, summary: List[Dict]):
        """Save test results summary"""
        summary_path = os.path.join(self.results_dir, "summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Batch Recognition Test Results\n")
            f.write("=============================\n\n")
            
            for entry in summary:
                f.write(f"Image: {entry['filename']}\n")
                for result in entry['results']:
                    if result['recognized']:
                        f.write(f"- Student {result['student_id']}\n")
                        f.write(f"  Confidence: {result['confidence']:.2f}\n")
                    else:
                        f.write("- Unknown person\n")
                        f.write(f"  Confidence: {result['confidence']:.2f}\n")
                f.write("\n")

def main():
    # Create test images folder if it doesn't exist
    test_folder = os.path.join("src", "test", "test_images")
    os.makedirs(test_folder, exist_ok=True)
    
    if not os.listdir(test_folder):
        print(f"\nError: No images found in {test_folder}")
        print("Please add test images to the folder and try again.")
        return
    
    # Initialize and run batch tester
    tester = BatchFaceRecognitionTester()
    tester.process_test_folder(test_folder)

if __name__ == "__main__":
    main() 