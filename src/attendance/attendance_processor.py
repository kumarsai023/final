import numpy as np
import os
import cv2
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import csv

class AttendanceProcessor:
    def __init__(self, threshold: float = 0.7):
        """Initialize attendance processor"""
        self.threshold = threshold
        
        # Create results directory
        self.results_dir = os.path.join("src", "attendance", "results")
        self.reports_dir = os.path.join(self.results_dir, "attendance_reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Load global model
        self.load_latest_global_model()
    
    def load_latest_global_model(self):
        """Load the latest global model"""
        global_model_dir = os.path.join("data", "global_model")
        model_files = [f for f in os.listdir(global_model_dir) 
                      if f.startswith('global_model_')]
        
        if not model_files:
            raise Exception("No global model found!")
        
        latest_model = os.path.join(global_model_dir, sorted(model_files)[-1])
        self.model_data = np.load(latest_model, allow_pickle=True)
        
        # Load student embeddings
        self.student_embeddings = {}
        student_ids = self.model_data['student_ids']
        for student_id in student_ids:
            self.student_embeddings[str(student_id)] = self.model_data[str(student_id)]
    
    def process_attendance(
        self,
        face_data: List[Dict],
        photo_path: str
    ) -> Tuple[List[Dict], str]:
        """
        Process attendance for detected faces
        Returns: List of attendance results and path to report
        """
        attendance_results = []
        present_students = set()
        
        # Process each face
        for face in face_data:
            embedding = face['embedding']
            best_match = self._find_best_match(embedding)
            
            if best_match['similarity'] >= self.threshold:
                present_students.add(best_match['student_id'])
                result = {
                    'recognized': True,
                    'student_id': best_match['student_id'],
                    'confidence': best_match['similarity']
                }
            else:
                result = {
                    'recognized': False,
                    'student_id': None,
                    'confidence': best_match['similarity']
                }
            
            attendance_results.append(result)
        
        # Generate report
        report_path = self._generate_report(
            attendance_results,
            present_students,
            photo_path
        )
        
        return attendance_results, report_path
    
    def _find_best_match(self, embedding: np.ndarray) -> Dict:
        """Find best matching student for a face embedding"""
        best_match = {
            'student_id': None,
            'similarity': 0.0
        }
        
        for student_id, stored_embedding in self.student_embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, stored_embedding) / \
                        (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
            
            if similarity > best_match['similarity']:
                best_match = {
                    'student_id': student_id,
                    'similarity': similarity
                }
        
        return best_match
    
    def _generate_report(
        self,
        attendance_results: List[Dict],
        present_students: set,
        photo_path: str
    ) -> str:
        """Generate attendance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(
            self.reports_dir,
            f"attendance_report_{timestamp}.csv"
        )
        
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attendance Report', timestamp])
            writer.writerow(['Photo', photo_path])
            writer.writerow([])
            writer.writerow(['Present Students'])
            
            for student_id in sorted(present_students):
                writer.writerow([f'Student {student_id}'])
            
            writer.writerow([])
            writer.writerow(['Detailed Results'])
            writer.writerow(['Face', 'Status', 'Student ID', 'Confidence'])
            
            for idx, result in enumerate(attendance_results):
                writer.writerow([
                    f'Face {idx+1}',
                    'Recognized' if result['recognized'] else 'Unknown',
                    result['student_id'] if result['recognized'] else 'N/A',
                    f"{result['confidence']:.2f}"
                ])
        
        return report_path

def main():
    """Test the attendance processor"""
    from photo_processor import PhotoProcessor
    
    # Initialize processors
    photo_processor = PhotoProcessor()
    attendance_processor = AttendanceProcessor()
    
    # Get test photo
    while True:
        photo_path = input("\nEnter attendance photo path (or 'q' to quit): ").strip().strip('"').strip("'")
        if photo_path.lower() == 'q':
            return
            
        if os.path.exists(photo_path):
            break
        else:
            print(f"Error: Photo not found at: {photo_path}")
            print("Please enter a valid path or 'q' to quit")
    
    try:
        # Process photo
        print("\nProcessing photo...")
        face_data, processed_path = photo_processor.process_photo(photo_path)
        print(f"\nDetected {len(face_data)} faces")
        
        # Process attendance
        print("\nProcessing attendance...")
        attendance_results, report_path = attendance_processor.process_attendance(
            face_data,
            photo_path
        )
        
        # Mark attendance results on photo
        print("\nMarking results...")
        final_path = photo_processor.mark_attendance_results(
            processed_path,
            face_data,
            attendance_results
        )
        
        print("\nAttendance Processing Complete!")
        print(f"Processed photo: {final_path}")
        print(f"Attendance report: {report_path}")
        
    except Exception as e:
        print(f"Error processing attendance: {str(e)}")

if __name__ == "__main__":
    main() 