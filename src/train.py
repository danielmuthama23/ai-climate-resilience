import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_hybrid_model

def train_model(data_path: str, model_save_path: str):
    """Train the hybrid model with early stopping and checkpointing."""
    # Load data
    X = np.load(os.path.join(data_path, "X.npy"))
    y = np.load(os.path.join(data_path, "y.npy"))
    
    # Split data (assuming X contains both spatial/temporal data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Build and compile model
    model = build_hybrid_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # Train
    history = model.fit(
        [X_train[:, :224*224*3].reshape(-1, 224, 224, 3), X_train[:, 224*224*3:].reshape(-1, 10, 2)],
        y_train,
        validation_data=([X_val[:, :224*224*3].reshape(-1, 224, 224, 3), X_val[:, 224*224*3:].reshape(-1, 10, 2)], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
if __name__ == "__main__":
    # Example: python train.py --data_path ./data/processed --model_save_path ./models/trained_model.h5
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_save_path", type=str)
    args = parser.parse_args()
    train_model(args.data_path, args.model_save_path)