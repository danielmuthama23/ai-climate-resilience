import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_satellite_data(sat_dir: str) -> np.ndarray:
    """Load and flatten satellite imagery from GeoTIFF files."""
    satellite_images = []
    for filename in os.listdir(sat_dir):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(sat_dir, filename)) as src:
                img = src.read(1)  # Read first band
                satellite_images.append(img.flatten())
    return np.array(satellite_images)

def preprocess_sensor_data(sensor_path: str, window_size: int = 10) -> tuple:
    """Process IoT sensor data into time sequences."""
    df = pd.read_csv(sensor_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['rainfall', 'soil_moisture']])
    
    # Create time-series sequences
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(df['flood_occurred'].iloc[i+window_size])
    
    return np.array(X), np.array(y), scaler

def save_processed_data(X: np.ndarray, y: np.ndarray, save_dir: str):
    """Save processed data as numpy arrays."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

if __name__ == "__main__":
    # Example usage: python data_pipeline.py --raw_path ./data/raw --save_path ./data/processed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    
    # Process data
    satellite_data = load_satellite_data(os.path.join(args.raw_path, "satellite"))
    X_sensor, y, scaler = preprocess_sensor_data(os.path.join(args.raw_path, "sensor_data.csv"))
    X = np.concatenate([satellite_data[:len(X_sensor)], axis=1)  # Align spatial/temporal data
    save_processed_data(X, y, args.save_path)