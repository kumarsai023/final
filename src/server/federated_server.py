import socket
import json
import threading
import logging
import os
import sys
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.protocol import FederatedProtocol, Message, MessageType

class FederatedServer:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.server_socket = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Store student embeddings as dictionary with student IDs as keys
        self.global_model_dir = "data/global_model"
        os.makedirs(self.global_model_dir, exist_ok=True)
        self.student_embeddings = {}  # Format: {'student_01': embedding1, 'student_02': embedding2, ...}
    
    def send_message(self, client_socket: socket.socket, message: Message):
        """Send message with length prefix"""
        data = message.to_json().encode()
        message_length = len(data)
        client_socket.send(str(message_length).zfill(10).encode())
        client_socket.send(data)
    
    def receive_message(self, client_socket: socket.socket) -> Message:
        """Receive message with length prefix"""
        message_length = int(client_socket.recv(10).decode())
        message_data = client_socket.recv(message_length).decode()
        return Message.from_json(message_data)
    
    def start(self):
        """Start the server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.logger.info(f"Server started on {self.host}:{self.port}")
            self.logger.info("Waiting for student connections...")
            
            while True:
                client_socket, address = self.server_socket.accept()
                self.logger.info(f"New connection from {address}")
                
                # Handle client in a new thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                client_thread.start()
                
        except KeyboardInterrupt:
            self.logger.info("Server shutting down...")
        except Exception as e:
            self.logger.error(f"Server error: {str(e)}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def handle_client(self, client_socket: socket.socket):
        """Handle client connection"""
        try:
            # Receive message
            message = self.receive_message(client_socket)
            
            if message.type == MessageType.CONNECT:
                student_id = message.data["student_id"]
                self.logger.info(f"Student {student_id} connected")
                
                # Send acknowledgment
                ack = FederatedProtocol.create_ack_message(
                    success=True,
                    message=f"Connected successfully"
                )
                self.send_message(client_socket, ack)
                
                # Receive features
                message = self.receive_message(client_socket)
                
                if message.type == MessageType.SEND_FEATURES:
                    features = np.array(message.data["features"])
                    student_id = message.data["student_id"]
                    
                    # Store features
                    self.student_embeddings[student_id] = features
                    self.save_global_model()
                    
                    self.logger.info(f"Received and saved features from {student_id}")
                    
                    # Send acknowledgment
                    ack = FederatedProtocol.create_ack_message(
                        success=True,
                        message="Features received and saved successfully"
                    )
                    self.send_message(client_socket, ack)
            
        except Exception as e:
            self.logger.error(f"Error handling client: {str(e)}")
        finally:
            client_socket.close()
    
    def save_global_model(self):
        """Save current global model with student IDs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.global_model_dir,
            f"global_model_{timestamp}.npz"
        )
        
        # Save as a dictionary with student IDs
        np.savez(
            save_path,
            **self.student_embeddings,  # Save each student's embedding with their ID
            student_ids=list(self.student_embeddings.keys()),  # Save list of student IDs
            timestamp=timestamp
        )
        
        self.logger.info(f"Global model saved with {len(self.student_embeddings)} students")
        self.logger.info(f"Student IDs in model: {list(self.student_embeddings.keys())}")

    def load_global_model(self, model_path: str = None):
        """Load the latest global model"""
        if model_path is None:
            # Get the latest model file
            model_files = [f for f in os.listdir(self.global_model_dir) if f.startswith('global_model_')]
            if not model_files:
                return
            model_path = os.path.join(self.global_model_dir, sorted(model_files)[-1])
        
        # Load the model
        data = np.load(model_path, allow_pickle=True)
        
        # Load student IDs and their embeddings
        student_ids = data['student_ids']
        self.student_embeddings = {}
        for student_id in student_ids:
            self.student_embeddings[str(student_id)] = data[str(student_id)]
        
        self.logger.info(f"Loaded global model with {len(self.student_embeddings)} students")
        self.logger.info(f"Student IDs: {list(self.student_embeddings.keys())}")

    def get_student_embedding(self, student_id: str) -> np.ndarray:
        """Get embedding for a specific student"""
        return self.student_embeddings.get(student_id)

    def list_registered_students(self):
        """List all registered students"""
        return list(self.student_embeddings.keys())

def main():
    """Run the server"""
    print("\nStarting Federated Server")
    print("=======================")
    server = FederatedServer()
    server.start()

if __name__ == "__main__":
    main() 