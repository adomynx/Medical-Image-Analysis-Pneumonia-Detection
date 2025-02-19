import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Preprocess an image by resizing and normalizing.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_dataset(data_dir, target_size=(150, 150)):
    """
    Load and preprocess the dataset.
    """
    images = []
    labels = []
    for label in ["NORMAL", "PNEUMONIA"]:
        class_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = preprocess_image(image_path, target_size)
            images.append(image)
            labels.append(0 if label == "NORMAL" else 1)
    return np.array(images), np.array(labels)