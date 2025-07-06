Task-03: SVM for Cat vs Dog Classification

🧰 Tools Required
	•	Python 3
	•	Libraries: numpy, pandas, matplotlib, scikit-learn, opencv or PIL

🔶 Step-by-Step Implementation

1. 📦 Dataset Preparation

Download the dataset from Kaggle Dogs vs Cats Dataset. After downloading:
	•	Unzip the file.
	•	You’ll get two folders: train/ and test/, each containing images like:
	•	cat.0.jpg, dog.1.jpg, etc.

 2. 📥 Load and Preprocess Images

Due to high resolution and count of images, SVM might struggle. So, we’ll:
	•	Resize images (e.g. to 64x64)
	•	Flatten into vectors (4096 features for 64x64x1)
	•	Normalize pixel values

 🔚 Summary
	•	SVM is effective for small image sets.
	•	Preprocessing (resizing, grayscale, normalization) is key.
	•	For better performance on full dataset, consider:
	•	PCA to reduce dimensionality
	•	Using LinearSVC for large-scale linear classification
	•	Deep Learning models (like CNNs) for higher accuracy
