import cv2
import numpy as np
import os
import sys
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attendance.attendance_checker import AttendanceChecker

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
        global_model_dir = "data/global_model"
        model_files = [f for f in os.listdir(global_model_dir) 
                      if f.startswith('global_model_')]
        
        if not model_files:
            raise Exception("No global model found!")
        
        latest_model = os.path.join(global_model_dir, sorted(model_files)[-1])
        print(f"\nLoading global model: {latest_model}")
        self.attendance_checker = AttendanceChecker(latest_model)
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single image and return results"""
        results = []
        
        # Get faces
        faces = self.app.get(image)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Check against global model
            match_result = self.attendance_checker.mark_attendance(
                embedding,
                threshold=0.7
            )
            
            # Determine color and text based on match
            if "Present: Student" in match_result:
                color = self.GREEN
                student_id = match_result.split("Student ")[1].split(" ")[0]
                text = f"Student {student_id}"
                confidence = float(match_result.split("Confidence: ")[1].strip(")"))
            else:
                color = self.RED
                text = "Unknown"
                confidence = 0.0
            
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
                'match_result': match_result,
                'confidence': confidence,
                'recognized': color == self.GREEN
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
                    print(f"- {result['match_result']}")
        
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
                    f.write(f"- {result['match_result']}\n")
                    f.write(f"  Confidence: {result['confidence']:.2f}\n")
                    f.write(f"  Recognized: {result['recognized']}\n")
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