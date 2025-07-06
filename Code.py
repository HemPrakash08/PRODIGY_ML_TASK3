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

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

model = SVC(kernel='linear')  # or 'rbf'
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

import matplotlib.pyplot as plt

for i in range(5):
    img = X_test[i].reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title("Predicted: " + ("Dog" if y_pred[i] else "Cat"))
    plt.axis('off')
    plt.show()
