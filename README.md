Dataset
The dataset used in this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. It contains 5,856 chest X-ray images divided into two categories:
- Normal: 1,583 images
- Pneumonia: 4,273 images

data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/


pip install -r requirements.txt

Data Augmentation: Use data augmentation techniques to improve model generalization.
Transfer Learning: Use a pre-trained model like VGG16 or ResNet for better performance.
Class Imbalance: Address the class imbalance in the dataset using techniques like oversampling or weighted loss.
Hyperparameter Tuning: Experiment with different hyperparameters (e.g., learning rate, batch size) to improve accuracy.

