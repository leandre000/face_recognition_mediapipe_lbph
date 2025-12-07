import cv2
import os
import json
import numpy as np

DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

images = []
labels = []
label_map = {}
current_label = 0

# Load dataset
for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        images.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

# Train LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(images, labels)

# Save model
recognizer.save(f"{MODEL_DIR}/lbph_model.xml")

# Save label map
with open(f"{MODEL_DIR}/label_map.json", "w") as f:
    json.dump(label_map, f)

print("\nTraining completed!")
print("Model saved to models/lbph_model.xml")
print("Label map saved to models/label_map.json")
