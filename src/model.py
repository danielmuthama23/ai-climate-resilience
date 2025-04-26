from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.applications import ResNet50

def build_hybrid_model(use_resnet: bool = True, time_steps: int = 10, input_features: int = 2):
    """Build hybrid CNN (spatial) + LSTM (temporal) model."""
    # Spatial branch (satellite imagery)
    spatial_input = Input(shape=(224, 224, 3))
    if use_resnet:
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=spatial_input)
        spatial_features = Flatten()(base_model.output)
    else:
        x = Conv2D(32, (3, 3), activation='relu')(spatial_input)
        spatial_features = Flatten()(x)
    
    # Temporal branch (sensor data)
    temporal_input = Input(shape=(time_steps, input_features))
    lstm_out = LSTM(64)(temporal_input)
    
    # Fusion
    merged = concatenate([spatial_features, lstm_out])
    dense = Dense(128, activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(dense)
    
    return Model(inputs=[spatial_input, temporal_input], outputs=output)