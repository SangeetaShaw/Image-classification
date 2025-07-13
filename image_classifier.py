import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Set dataset path and image parameters
DATASET_DIR = "dataset"
IMAGE_SIZE = (128, 128)

def extract_features_and_labels(dataset_path):
    features = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = imread(img_path, as_gray=True)
                img_resized = resize(img, IMAGE_SIZE)
                hog_features = hog(img_resized, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2), orientations=9)
                features.append(hog_features)
                labels.append(label)
            except:
                print(f"Failed to process image: {img_path}")
    return np.array(features), np.array(labels), class_names

# Load features and labels
print("Extracting features...")
X, y, class_names = extract_features_and_labels(DATASET_DIR)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose model
print("Training model...")
model = SVC(kernel='linear')  # Or use RandomForestClassifier(), GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
from sklearn.utils.multiclass import unique_labels

labels = unique_labels(y_test, y_pred)
label_names = [class_names[i] for i in labels]
print(classification_report(y_test, y_pred, labels=labels, target_names=label_names))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
