import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, limit=1000, img_size=64):
    X = []
    y = []
    count = 0
    for filename in os.listdir(folder):
        if count >= limit:
            break
        label = 0 if 'cat' in filename else 1  # 0: cat, 1: dog
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
        img = cv2.resize(img, (img_size, img_size))
        X.append(img.flatten())
        y.append(label)
        count += 1
    return np.array(X), np.array(y)

X, y = load_images_from_folder('path_to_train_folder', limit=2000)
X = X / 255.0  # normalize
