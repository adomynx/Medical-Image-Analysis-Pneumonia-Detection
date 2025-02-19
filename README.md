## **Dataset**
The dataset used in this project is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. It contains **5,856 chest X-ray images** divided into two categories:
- **Normal**: 1,583 images
- **Pneumonia**: 4,273 images

The dataset is split into the following folder structure:

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
---

## **Install Dependencies**
To install the required dependencies, run:
```bash
pip install -r requirements.txt

