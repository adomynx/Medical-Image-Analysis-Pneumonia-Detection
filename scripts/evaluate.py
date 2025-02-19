import sys
import os
import numpy as np
from tensorflow.keras.models import load_model

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the utils module
from utils.preprocessing import load_dataset

# Load the test dataset
data_dir = "data/chest_xray/test"  # Update this path
images, labels = load_dataset(data_dir)

# Reshape images for CNN input
images = np.expand_dims(images, axis=-1)

# Load the trained model
model = load_model("models/pneumonia_detection_model.keras")

# Evaluate the model
loss, accuracy = model.evaluate(images, labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")