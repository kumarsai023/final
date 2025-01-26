# Federated Learning Face Recognition Attendance System

A facial recognition-based attendance system using federated learning to maintain student privacy and provide efficient attendance tracking.

## Project Structure
```plaintext
project_root/
├── data/
│   ├── embeddings/        # Student face embeddings
│   ├── global_model/      # Federated learning model
│   └── student_data/      # Individual student face data
│
└── src/
    ├── attendance/        # Attendance processing system
    │   ├── row_photos/    # Classroom photos for processing
    │   └── results/       # Processed results and reports
    │
    ├── client/           # Federated learning client
    ├── data_collection/  # Face data collection
    ├── feature_extraction/ # Embedding generation
    ├── server/          # Federated learning server
    ├── shared/          # Shared utilities
    └── test/            # Testing utilities
```

## System Requirements
- Python 3.8 or higher
- OpenCV
- NumPy
- InsightFace
- ONNX Runtime

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required packages:
```bash
pip install opencv-python numpy insightface onnxruntime
```

3. Create necessary directories:
```bash
mkdir -p data/embeddings data/global_model data/student_data
mkdir -p src/attendance/row_photos src/attendance/results
mkdir -p src/test/test_images
```

## Usage

### 1. Data Collection
Collect student face data:
```bash
python src/data_collection/face_collector.py
```
- Enter student number when prompted
- Follow camera instructions
- Collected images will be saved in `data/student_data/`

### 2. Feature Extraction
Generate face embeddings:
```bash
python src/feature_extraction/face_processor.py
```
- Enter student number
- Embeddings will be saved in `data/embeddings/`

### 3. Federated Learning
a. Start the server:
```bash
python src/server/federated_server.py
```

b. Run the client to send embeddings:
```bash
python src/client/federated_client.py
```
- Enter student number
- Client will send embeddings to server
- Server updates global model

### 4. Testing Recognition
Test the global model:
```bash
python src/test/batch_recognition_test.py
```
- Add test images to `src/test/test_images/`
- Results will show recognized students

### 5. Attendance Processing
Process classroom attendance:
```bash
python src/attendance/attendance_processor.py
```
- Add classroom photos to `src/attendance/row_photos/`
- System will:
  - Identify students
  - Mark attendance
  - Generate reports

## Features
- Privacy-preserving federated learning
- Real-time face detection and recognition
- Batch processing of classroom photos
- Detailed attendance reports
- Visual feedback with marked faces
- Support for multiple classroom photos

## File Descriptions
- `face_collector.py`: Collects student face data
- `face_processor.py`: Generates face embeddings
- `federated_server.py`: Manages global model
- `federated_client.py`: Sends student data to server
- `attendance_processor.py`: Processes attendance
- `batch_recognition_test.py`: Tests recognition system

## Output Files
1. **Processed Photos**
   - Green boxes: Recognized students
   - Red boxes: Unknown faces
   - Student IDs displayed

2. **Attendance Reports**
   - Date and time
   - Present students
   - Confidence scores
   - Photo-wise details

## Best Practices
1. **Data Collection**
   - Good lighting conditions
   - Multiple angles
   - Clear face visibility

2. **Attendance Photos**
   - Clear classroom photos
   - Proper lighting
   - Multiple row-wise photos if needed

3. **System Usage**
   - Regular model updates
   - Periodic testing
   - Backup of important data

## Troubleshooting
1. **No Faces Detected**
   - Check image quality
   - Ensure proper lighting
   - Verify file format

2. **Recognition Issues**
   - Update global model
   - Check threshold settings
   - Verify student data

3. **System Errors**
   - Check directory structure
   - Verify file permissions
   - Ensure all dependencies are installed

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request



## Contact
kumarsaiofficial@gmail.com