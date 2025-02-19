import sys
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the utils module
from utils.preprocessing import preprocess_image

# Function to get a random image from the dataset
def get_random_image(data_dir):
    # Choose a random class (NORMAL or PNEUMONIA)
    class_name = random.choice(["NORMAL", "PNEUMONIA"])
    class_dir = os.path.join(data_dir, class_name)
    
    # Choose a random image from the selected class
    image_name = random.choice(os.listdir(class_dir))
    image_path = os.path.join(class_dir, image_name)
    
    return image_path, class_name

# Path to the test dataset
data_dir = "data/chest_xray/test"  # Update this path

# Get a random image and its true label
image_path, true_label = get_random_image(data_dir)

# Debug: Print the selected image and true label
print(f"Selected image: {image_path}")
print(f"True label: {true_label}")

# Load the trained model
model = load_model("models/pneumonia_detection_model.keras")

# Preprocess the image
try:
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)

    # Make a prediction
    prediction = model.predict(image)
    predicted_label = "Pneumonia" if prediction > 0.5 else "Normal"
    
    # Print the prediction
    print(f"Model's prediction: {predicted_label}")

    # Display the selected image
    image_display = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB))
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()
except Exception as e:
    print(f"Error: {e}")