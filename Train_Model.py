import os
import cv2
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "Dataset/asl_alphabet_train/asl_alphabet_train"

IMG_SIZE = (64, 64)

# Start small for testing
MAX_IMAGES_PER_CLASS = 100


# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================

def extract_hog(image):

    image = cv2.resize(image, IMG_SIZE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")

X = []
y = []

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Classes:", classes)
print("Number of classes:", len(classes))


for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    image_files = [
        file for file in os.listdir(class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Limit number of images for first training
    image_files = image_files[:MAX_IMAGES_PER_CLASS]

    print(f"Processing {class_name}: {len(image_files)} images")

    for image_file in image_files:

        image_path = os.path.join(class_path, image_file)

        image = cv2.imread(image_path)

        if image is None:
            continue

        features = extract_hog(image)

        X.append(features)
        y.append(class_name)


print("\nDataset loading complete!")

print("Total samples:", len(X))


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# TRAIN SVM
# ==========================================

print("\nTraining SVM model...")
print("This may take some time...")


model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)


print("SVM training complete!")


# ==========================================
# EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("MODEL ACCURACY:", accuracy)
print("================================")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

model_data = {
    "model": model,
    "classes": classes,
    "img_size": IMG_SIZE
}

joblib.dump(
    model_data,
    "asl_model.pkl"
)

print("\n================================")
print("MODEL SAVED!")
print("File: asl_model.pkl")
print("================================")