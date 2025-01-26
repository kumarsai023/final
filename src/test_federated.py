import subprocess
import time
import sys
import os
import threading

def print_output(process, name):
    """Print process output in real-time"""
    for line in process.stdout:
        print(f"{name}: {line.strip()}")

def run_server():
    """Run the federated server"""
    print("\nStarting Federated Server...")
    server_process = subprocess.Popen(
        [sys.executable, "src/server/federated_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        text=True  # This handles the decoding automatically
    )
    # Start output thread
    server_thread = threading.Thread(
        target=print_output, 
        args=(server_process, "Server")
    )
    server_thread.daemon = True
    server_thread.start()
    
    time.sleep(2)  # Wait for server to start
    return server_process

def run_client():
    """Run the federated client"""
    print("\nStarting Federated Client...")
    client_process = subprocess.Popen(
        [sys.executable, "src/client/federated_client.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        text=True  # This handles the decoding automatically
    )
    # Start output thread
    client_thread = threading.Thread(
        target=print_output, 
        args=(client_process, "Client")
    )
    client_thread.daemon = True
    client_thread.start()
    
    return client_process

def main():
    """Test the complete federated system"""
    print("\nFederated Learning System Test")
    print("============================")
    
    # Check for student embedding
    if not os.path.exists("data/embeddings/student_embedding.npy"):
        print("\nError: No student embedding found!")
        print("Please run feature extraction first.")
        return
    
    try:
        # Start server
        server_process = run_server()
        print("Server started successfully")
        
        # Start client
        client_process = run_client()
        print("Client started successfully")
        
        # Wait for processes to complete
        time.sleep(10)  # Give more time for communication
        
    except KeyboardInterrupt:
        print("\nStopping test...")
    
    finally:
        # Cleanup
        print("\nCleaning up processes...")
        server_process.terminate()
        client_process.terminate()
        print("\nTest complete!")

if __name__ == "__main__":
    main() 