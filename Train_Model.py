import os
import cv2
import joblib
import mediapipe as mp
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "Dataset/asl_alphabet_train/asl_alphabet_train"

IMG_SIZE = (64, 64)

MAX_IMAGES_PER_CLASS = 1000

PADDING = 50

MODEL_OUTPUT = "asl_model_mediapipe.pkl"


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ============================================================
# HOG
# ============================================================

def extract_hog(image):

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ============================================================
# MEDIAPIPE HAND CROP
# ============================================================

def get_hand_crop(image):

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    xs = []
    ys = []

    for landmark in hand.landmark:

        xs.append(
            int(landmark.x * width)
        )

        ys.append(
            int(landmark.y * height)
        )

    x_min = max(
        0,
        min(xs) - PADDING
    )

    y_min = max(
        0,
        min(ys) - PADDING
    )

    x_max = min(
        width,
        max(xs) + PADDING
    )

    y_max = min(
        height,
        max(ys) + PADDING
    )

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[
        y_min:y_max,
        x_min:x_max
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# LOAD CLASSES
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(
            DATASET_PATH,
            folder
        )
    )
])


print()
print("==========================================")
print("MEDIAPIPE + HOG MODEL TRAINING")
print("==========================================")

print()
print("Dataset:", DATASET_PATH)
print("Classes:", len(classes))
print(classes)
print()


# ============================================================
# DATA
# ============================================================

X = []
y = []

total_images = 0
total_skipped = 0


# ============================================================
# PROCESS DATASET
# ============================================================

for class_index, class_name in enumerate(classes):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    image_files = [
        f
        for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files = image_files[
        :MAX_IMAGES_PER_CLASS
    ]

    print(
        f"[{class_index + 1:02d}/{len(classes):02d}] "
        f"{class_name}: "
        f"{len(image_files)} images"
    )

    class_success = 0

    for filename in image_files:

        image_path = os.path.join(
            class_path,
            filename
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            total_skipped += 1
            continue


        # ====================================================
        # SPECIAL CASE: NOTHING
        # ====================================================

        if class_name == "nothing":

            crop = image


        else:

            crop = get_hand_crop(
                image
            )

            if crop is None:

                total_skipped += 1
                continue


        # ====================================================
        # HOG
        # ====================================================

        features = extract_hog(
            crop
        )

        X.append(
            features
        )

        y.append(
            class_index
        )

        class_success += 1

        total_images += 1


    print(
        f"      usable: {class_success}"
    )


# ============================================================
# CLOSE MEDIAPIPE
# ============================================================

hands.close()


# ============================================================
# NUMPY
# ============================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)


print()
print("==========================================")
print("DATASET SUMMARY")
print("==========================================")

print(
    "Usable images:",
    len(X)
)

print(
    "Skipped images:",
    total_skipped
)

print(
    "Feature size:",
    X.shape[1]
)


# ============================================================
# CLASS COUNTS
# ============================================================

print()
print("Images per class:")

for class_index, class_name in enumerate(classes):

    count = np.sum(
        y == class_index
    )

    print(
        f"{class_name:8s}: {count}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()
print(
    "Training images:",
    len(X_train)
)

print(
    "Testing images:",
    len(X_test)
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Training LinearSVC...")


model = LinearSVC(

    C=10,

    max_iter=5000,

    dual="auto",

    random_state=42
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("==========================================")
print("RESULT")
print("==========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# REPORT
# ============================================================

print()

print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(

    {
        "model": model,
        "classes": classes,
        "img_size": IMG_SIZE,
        "padding": PADDING
    },

    MODEL_OUTPUT
)


print()
print("==========================================")
print("MODEL SAVED")
print("==========================================")

print(
    MODEL_OUTPUT
)

print()
print("Training complete.")