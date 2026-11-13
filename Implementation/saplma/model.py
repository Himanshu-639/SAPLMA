"""
SAPLMA Classifier Model Architecture
Strictly follows Azaria & Mitchell (EMNLP Findings 2023), Section 4:
"SAPLMA's classifier employs a feedforward neural network, featuring three hidden layers
with decreasing numbers of hidden units (256, 128, 64), all utilizing ReLU activations.
The final layer is a sigmoid output. We use the Adam optimizer. We do not fine-tune any
of these hyper-parameters for this task. The classifier is trained for 5 epochs."
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

def build_saplma_classifier(input_dim: int) -> Sequential:
    """
    Constructs the exact 3-hidden-layer MLP architecture described in Section 4.
    Input Dimension -> Dense(256, relu) -> Dense(128, relu) -> Dense(64, relu) -> Dense(1, sigmoid)
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def parse_embedding_str(val_str: str) -> np.ndarray:
    """Utility to safely parse serialized string embeddings into numpy float arrays."""
    clean_str = (
        val_str.replace("[array(", "")
               .replace("dtype=float32)]", "")
               .replace("\n", "")
               .replace(" ", "")
               .replace("],", "]")
               .replace("[", "")
               .replace("]", "")
    )
    return np.fromstring(clean_str, sep=',')
