import cv2
import numpy as np
import os
import sys
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import csv

class AttendanceProcessor:
    def __init__(self, threshold: float = 0.7):
        """Initialize attendance processor"""
        # Initialize InsightFace
        self.app = FaceAnalysis(
            name="buffalo_l",
            root="models",
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Set threshold for face recognition
        self.threshold = threshold
        
        # Colors for visualization
        self.GREEN = (0, 255, 0)  # BGR format
        self.RED = (0, 0, 255)    # BGR format
        
        # Create results directories
        self.results_dir = os.path.join("src", "attendance", "results")
        self.processed_dir = os.path.join(self.results_dir, "processed_photos")
        self.reports_dir = os.path.join(self.results_dir, "attendance_reports")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load global model
        self.load_latest_global_model()
        
        print("\nAttendance Processing System")
        print("==========================")
    
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
    
    def check_face(self, face_embedding: np.ndarray) -> Tuple[bool, str, float]:
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
        
        is_match = best_match['similarity'] >= self.threshold
        return is_match, best_match['student_id'], best_match['similarity']
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single image and return results"""
        results = []
        present_students = set()
        
        # Get faces
        faces = self.app.get(image)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Check against global model
            is_match, student_id, confidence = self.check_face(embedding)
            
            # Update present students
            if is_match:
                present_students.add(student_id)
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
        
        return image, results, present_students
    
    def process_attendance(self, row_photos_dir: str) -> Dict:
        """Process attendance for multiple photos"""
        # Supported image formats
        image_extensions = ('.jpg', '.jpeg', '.png')
        
        # Track attendance across all photos
        all_present_students = set()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.processed_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)
        
        # Results for report
        session_results = {
            'timestamp': timestamp,
            'photos_processed': [],
            'present_students': set(),
            'processed_photo_paths': []
        }
        
        # Process each photo
        for filename in os.listdir(row_photos_dir):
            if filename.lower().endswith(image_extensions):
                print(f"\nProcessing: {filename}")
                
                # Load image
                image_path = os.path.join(row_photos_dir, filename)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Error loading image: {filename}")
                    continue
                
                # Process image
                processed_image, results, present_students = self.process_image(image)
                
                # Update overall attendance
                all_present_students.update(present_students)
                
                # Save processed image
                output_path = os.path.join(session_dir, f"marked_{filename}")
                cv2.imwrite(output_path, processed_image)
                
                # Update session results
                session_results['photos_processed'].append({
                    'filename': filename,
                    'results': results,
                    'present_students': list(present_students)
                })
                session_results['processed_photo_paths'].append(output_path)
                
                print(f"Processed photo saved: marked_{filename}")
                print("Students detected:", ', '.join(f"Student {id}" for id in present_students) if present_students else "None")
        
        session_results['present_students'] = all_present_students
        
        # Generate attendance report
        report_path = self.generate_report(session_results)
        session_results['report_path'] = report_path
        
        return session_results
    
    def generate_report(self, session_results: Dict) -> str:
        """Generate attendance report"""
        timestamp = session_results['timestamp']
        report_path = os.path.join(self.reports_dir, f"attendance_report_{timestamp}.csv")
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attendance Report', timestamp])
            writer.writerow([])
            
            # Overall attendance
            writer.writerow(['Present Students'])
            for student_id in sorted(session_results['present_students']):
                writer.writerow([f'Student {student_id}'])
            
            writer.writerow([])
            writer.writerow(['Detailed Results by Photo'])
            
            # Details for each photo
            for photo_data in session_results['photos_processed']:
                writer.writerow([f"\nPhoto: {photo_data['filename']}"])
                writer.writerow(['Face', 'Status', 'Student ID', 'Confidence'])
                
                for idx, result in enumerate(photo_data['results']):
                    writer.writerow([
                        f'Face {idx+1}',
                        'Present' if result['recognized'] else 'Unknown',
                        result['student_id'] if result['recognized'] else 'N/A',
                        f"{result['confidence']:.2f}"
                    ])
        
        return report_path

def main():
    """Run attendance processing"""
    # Initialize processor
    processor = AttendanceProcessor()
    
    # Get row photos directory
    row_photos_dir = os.path.join("src", "attendance", "row_photos")
    os.makedirs(row_photos_dir, exist_ok=True)
    
    if not os.listdir(row_photos_dir):
        print(f"\nError: No photos found in {row_photos_dir}")
        print("Please add classroom photos to this directory and try again.")
        print("\nAccepted formats: .jpg, .jpeg, .png")
        return
    
    try:
        # Process attendance
        results = processor.process_attendance(row_photos_dir)
        
        print("\nAttendance Processing Complete!")
        print(f"Processed {len(results['photos_processed'])} photos")
        print(f"Total students present: {len(results['present_students'])}")
        print(f"\nProcessed photos saved in: {os.path.dirname(results['processed_photo_paths'][0])}")
        print(f"Attendance report saved as: {results['report_path']}")
        
    except Exception as e:
        print(f"Error processing attendance: {str(e)}")

if __name__ == "__main__":
    main() 