import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = "./sign_data"
CLASSES = ["A", "B", "C", "HELLO"]

data = []
labels = []

print("Loading dataset...")
for cls in CLASSES:
  class_dir = os.path.join(DATA_DIR, cls)
  for file_name in os.listdir(class_dir):
    if file_name.endswith(".npy"):
      file_path = os.path.join(class_dir, file_name)
      landmark_data = np.load(file_path)
      data.append(landmark_data)
      labels.append(cls)

X = np.array(data)
y = np.array(labels)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Random Forest model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save the trained model to disk
joblib.dump(model, "sign_model.p")
print("Model saved successfully as 'sign_model.p'!")