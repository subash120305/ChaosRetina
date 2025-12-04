
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import RFMiDDataset
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def verify_data():
    config = load_config('config/config.yaml')
    
    # Paths
    train_csv = config['paths']['train_labels']
    train_dir = config['paths']['train_images']
    
    print(f"Checking dataset at: {train_dir}")
    print(f"CSV: {train_csv}")
    
    # Load dataset
    dataset = RFMiDDataset(
        csv_path=train_csv,
        image_dir=train_dir,
        mode="multilabel",
        disease_columns=config['dataset']['disease_columns']
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Check label distribution
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label.numpy())
        if i >= 100: break  # Check first 100 samples
        
    labels = np.array(labels)
    
    print("\nLabel Stats (First 1000 samples):")
    print(f"Shape: {labels.shape}")
    print(f"Min: {labels.min()}, Max: {labels.max()}")
    print(f"Mean: {labels.mean()}")
    
    # Check per-class positives
    class_counts = labels.sum(axis=0)
    print("\nPer-class positives:")
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {int(count)}")
        
    if labels.sum() == 0:
        print("\n[CRITICAL] NO POSITIVE LABELS FOUND! The model cannot learn.")
    else:
        print("\n[OK] Positive labels found.")

if __name__ == "__main__":
    verify_data()
