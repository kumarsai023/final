import numpy as np
import matplotlib.pyplot as plt
import os
from aggregator import FeatureAggregator

def visualize_embeddings():
    # Load original embedding
    original = np.load("data/embeddings/student_embedding.npy")
    
    # Find the latest global model file
    model_dir = "data/federated_model"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("global_model_")]
    if not model_files:
        print("No global model files found!")
        return
    
    # Get the most recent file
    latest_model = max(model_files)
    aggregated = np.load(os.path.join(model_dir, latest_model))
    
    plt.figure(figsize=(15, 5))
    
    # Plot first 50 dimensions as example
    plt.subplot(1, 2, 1)
    plt.plot(original[:50], label='Original')
    plt.plot(aggregated[:50], label='Aggregated')
    plt.title('First 50 Dimensions')
    plt.legend()
    
    # Plot distribution
    plt.subplot(1, 2, 2)
    plt.hist(original, bins=50, alpha=0.5, label='Original')
    plt.hist(aggregated, bins=50, alpha=0.5, label='Aggregated')
    plt.title('Value Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png')
    plt.close()
    
    print(f"\nUsed model file: {latest_model}")

def main():
    """Test the feature aggregator"""
    aggregator = FeatureAggregator()
    
    # Load and process features
    feature_path = os.path.join("data", "embeddings", "student_embedding.npy")
    if os.path.exists(feature_path):
        features = np.load(feature_path)
        print("\nEmbedding Dimensions:")
        print(f"Original features: {features.shape}")
        
        # Add clients
        aggregator.add_client_features("client1", features)
        aggregator.add_client_features("client2", features * 0.9)
        
        # Aggregate
        global_model = aggregator.aggregate_features(min_clients=2)
        
        if global_model is not None:
            print(f"Aggregated model: {global_model.shape}")
            print("\nNorms:")
            print(f"Original norm: {np.linalg.norm(features)}")
            print(f"Aggregated norm: {np.linalg.norm(global_model)}")
            
            # Visualize
            visualize_embeddings()
            print("\nVisualization saved as 'embedding_comparison.png'")

if __name__ == "__main__":
    main()