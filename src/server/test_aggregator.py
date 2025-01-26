import numpy as np
import os
from aggregator import FeatureAggregator
import matplotlib.pyplot as plt

def visualize_matches(matches, top_n=5):
    """Visualize top matches"""
    # Get top N matches
    top_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    students, scores = zip(*top_matches)
    plt.bar(students, scores)
    plt.title('Top Student Matches')
    plt.xlabel('Student ID')
    plt.ylabel('Similarity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('match_results.png')
    plt.close()

def main():
    """Test single student feature aggregation"""
    print("\nSingle Student Test")
    print("==================")
    
    # 1. Initialize aggregator
    aggregator = FeatureAggregator()
    
    # 2. Load student embedding
    feature_path = os.path.join("data", "embeddings", "student_embedding.npy")
    if not os.path.exists(feature_path):
        print("Error: No student embedding found!")
        return
    
    student_embedding = np.load(feature_path)
    print(f"\nLoaded student embedding shape: {student_embedding.shape}")
    
    # 3. Add student embedding
    if aggregator.add_student_embedding("student_01", student_embedding):
        print("Added student embedding successfully")
        
        # 4. Save model
        if aggregator.save_model():
            print("Saved student model successfully")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 