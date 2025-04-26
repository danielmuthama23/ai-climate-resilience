import shap
import numpy as np
import tensorflow as tf

def explain_model(model_path: str, data_path: str, sample_size: int = 100):
    """Generate SHAP explanations for model predictions."""
    model = tf.keras.models.load_model(model_path)
    X = np.load(os.path.join(data_path, "X.npy"))[:sample_size]
    
    # Create explainer
    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # Visualize feature importance
    shap.summary_plot(shap_values, X, feature_names=['satellite_features', 'rainfall', 'soil_moisture'])
    
if __name__ == "__main__":
    # Example: python explainability.py --model_path ./models/trained_model.h5 --data_path ./data/processed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    explain_model(args.model_path, args.data_path)