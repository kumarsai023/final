import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from photo_processor import PhotoProcessor
from attendance_processor import AttendanceProcessor

class BatchPhotoProcessor:
    def __init__(self):
        """Initialize batch photo processor"""
        # Initialize processors
        self.photo_processor = PhotoProcessor()
        self.attendance_processor = AttendanceProcessor()
        
        # Create directories
        self.base_dir = os.path.join("src", "attendance")
        self.row_photos_dir = os.path.join(self.base_dir, "row_photos")
        os.makedirs(self.row_photos_dir, exist_ok=True)
        
        print("\nBatch Photo Processor")
        print("===================")
    
    def process_row_photos(self) -> Tuple[List[Dict], str]:
        """Process all photos in row_photos directory"""
        # Get all images
        image_files = [f for f in os.listdir(self.row_photos_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            raise ValueError(f"No images found in {self.row_photos_dir}")
        
        all_face_data = []
        processed_paths = []
        
        # Process each photo
        print(f"\nProcessing {len(image_files)} photos...")
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(self.row_photos_dir, img_file)
            print(f"\nProcessing photo {idx}/{len(image_files)}: {img_file}")
            
            # Process photo
            face_data, processed_path = self.photo_processor.process_photo(img_path)
            print(f"Detected {len(face_data)} faces")
            
            all_face_data.extend(face_data)
            processed_paths.append(processed_path)
        
        # Process attendance for all faces
        print("\nProcessing attendance for all faces...")
        attendance_results, report_path = self.attendance_processor.process_attendance(
            all_face_data,
            "Multiple Row Photos"
        )
        
        # Mark results on each processed photo
        print("\nMarking attendance results...")
        start_idx = 0
        final_paths = []
        
        for processed_path, face_count in zip(
            processed_paths,
            [len(self.photo_processor.app.get(cv2.imread(p))) for p in processed_paths]
        ):
            # Get results for this photo's faces
            photo_results = attendance_results[start_idx:start_idx + face_count]
            photo_face_data = all_face_data[start_idx:start_idx + face_count]
            
            # Mark results
            final_path = self.photo_processor.mark_attendance_results(
                processed_path,
                photo_face_data,
                photo_results
            )
            final_paths.append(final_path)
            
            start_idx += face_count
        
        return attendance_results, report_path, final_paths

def main():
    """Test the batch processor"""
    processor = BatchPhotoProcessor()
    
    # Check if row_photos directory is empty
    if not os.listdir(processor.row_photos_dir):
        print(f"\nError: No photos found in {processor.row_photos_dir}")
        print("Please add photos to this directory and try again")
        print("\nAccepted formats: .jpg, .jpeg, .png")
        return
    
    try:
        # Process all photos
        results, report_path, final_paths = processor.process_row_photos()
        
        print("\nBatch Processing Complete!")
        print(f"Total faces processed: {len(results)}")
        print(f"Attendance report: {report_path}")
        print("\nProcessed photos:")
        for idx, path in enumerate(final_paths, 1):
            print(f"{idx}. {path}")
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")

if __name__ == "__main__":
    main() 