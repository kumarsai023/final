import cv2
import os
import numpy as np
from datetime import datetime
import logging
from typing import List, Tuple, Optional
import time

class FaceDataCollector:
    def __init__(self, 
                 save_dir: str = "data",
                 required_samples: int = 100,
                 face_size: Tuple[int, int] = (224, 224),
                 capture_delay: float = 0.8):  # Reduced to 0.8 seconds
        """
        Initialize the face data collector for a single student
        """
        self.save_dir = save_dir
        self.required_samples = required_samples
        self.face_size = face_size
        self.capture_delay = capture_delay
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Create directory if it doesn't exist
        student_dir = os.path.join(save_dir, "student_data")
        os.makedirs(student_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def collect_student_data(self) -> List[np.ndarray]:
        """
        Collect face data for a single student with guided instructions
        """
        face_data = []
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.logger.error("Error: Could not open camera")
            return face_data

        last_capture_time = time.time()
        image_counter = 1  # Counter for sequential image naming
        
        # Simplified poses for faster collection
        poses = [
            "Look straight",
            "Turn left",
            "Turn right",
            "Tilt up",
            "Tilt down"
        ]
        current_pose = 0
        samples_per_pose = 20

        print("\nStarting face data collection...")
        print(f"Collecting {samples_per_pose} images for each pose")
        print("Press 'q' to quit or 'c' for manual capture")
        
        while len(face_data) < self.required_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Simple progress display
            cv2.putText(frame, f"Pose: {poses[current_pose]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {image_counter}/{self.required_samples}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            current_time = time.time()
            if (current_time - last_capture_time) >= self.capture_delay or key == ord('c'):
                face_roi = self.detect_face(frame)
                
                if face_roi is not None:
                    face_data.append(face_roi)
                    
                    # Save image with sequential naming
                    image_path = os.path.join(
                        self.save_dir, 
                        "student_data",
                        f"face_{image_counter:03d}.jpg"  # face_001.jpg format
                    )
                    cv2.imwrite(image_path, face_roi)
                    
                    print(f"Captured: face_{image_counter:03d}.jpg")
                    last_capture_time = current_time
                    image_counter += 1
                    
                    # Update pose if needed
                    if len(face_data) % samples_per_pose == 0 and current_pose < len(poses) - 1:
                        current_pose += 1
                        print(f"\nNew pose: {poses[current_pose]}")
                        time.sleep(0.5)  # Reduced pose adjustment time
                
                else:
                    print("No face detected!")
            
            # Show face preview
            face_preview = self.detect_face(frame)
            if face_preview is not None:
                cv2.imshow('Face Preview', face_preview)
        
        cap.release()
        cv2.destroyAllWindows()
        
        return face_data

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, self.face_size)
        
        return face_roi

def main():
    """Main function to run the face data collection process"""
    collector = FaceDataCollector(
        save_dir="face_data",
        required_samples=100,
        face_size=(224, 224),
        capture_delay=0.8  # Faster capture rate
    )
    
    print("\nFace Data Collection System")
    print("===========================")
    face_data = collector.collect_student_data()
    
    if len(face_data) >= collector.required_samples:
        print("\nCollection completed!")
        print(f"Total images: {len(face_data)}")
    else:
        print("\nCollection incomplete")
        print(f"Collected: {len(face_data)}/{collector.required_samples}")

if __name__ == "__main__":
    main() 