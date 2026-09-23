import cv2
import os
import mediapipe as mp

# =========================
# SETTINGS
# =========================

DATASET_PATH = "Dataset/MyDataset"

IMG_SIZE = (128, 128)
MAX_IMAGES = 300

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

# =========================
# CREATE ALL FOLDERS
# =========================

os.makedirs(DATASET_PATH, exist_ok=True)

for class_name in CLASSES:
    os.makedirs(
        os.path.join(DATASET_PATH, class_name),
        exist_ok=True
    )

print("\nAll dataset folders are ready.")

print("\nAvailable classes:")
print(", ".join(CLASSES))

# =========================
# ASK FOR SIGN
# =========================

sign = input("\nEnter the sign you want to collect: ").strip()

if sign.upper() in CLASSES:
    sign = sign.upper()

if sign not in CLASSES:
    print("Invalid class name.")
    print("Please enter one of:", ", ".join(CLASSES))
    exit()

save_folder = os.path.join(DATASET_PATH, sign)

# =========================
# COUNT EXISTING IMAGES
# =========================

existing_images = [
    file for file in os.listdir(save_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

count = len(existing_images)

print(f"\nClass: {sign}")
print(f"Existing images: {count}")
print(f"Maximum images: {MAX_IMAGES}")

if count >= MAX_IMAGES:
    print(f"\n{sign} already has {MAX_IMAGES} images.")
    print("Choose another class.")
    exit()

# =========================
# MEDIAPIPE HAND DETECTOR
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("\nCamera started.")
print("Show your hand to the camera.")
print("Press SPACE to save an image.")
print("Press Q to quit.")

# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    hand_crop = None
    hand_box = None

    # =========================
    # DETECT HAND
    # =========================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        height, width, _ = frame.shape

        # Get landmark coordinates
        x_coordinates = [
            int(landmark.x * width)
            for landmark in hand_landmarks.landmark
        ]

        y_coordinates = [
            int(landmark.y * height)
            for landmark in hand_landmarks.landmark
        ]

        # Bounding box
        x_min = max(0, min(x_coordinates))
        y_min = max(0, min(y_coordinates))
        x_max = min(width, max(x_coordinates))
        y_max = min(height, max(y_coordinates))

        # Add padding around hand
        padding = 30

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        hand_box = (x_min, y_min, x_max, y_max)

        # Crop hand
        hand_crop = frame[
            y_min:y_max,
            x_min:x_max
        ]

        # Draw hand box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            "HAND DETECTED",
            (x_min, max(30, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # =========================
    # INFORMATION
    # =========================

    cv2.putText(
        frame,
        f"Sign: {sign}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Images: {count}/{MAX_IMAGES}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Save | Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("SignSpeak - Collect Dataset", frame)

    # =========================
    # KEYBOARD
    # =========================

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        if hand_crop is None or hand_crop.size == 0:

            print("No hand detected. Image NOT saved.")

            continue

        if count >= MAX_IMAGES:

            print(f"{sign} is complete.")

            break

        # Resize exactly to training size
        hand_image = cv2.resize(
            hand_crop,
            IMG_SIZE
        )

        count += 1

        filename = f"{sign}_{count:04d}.jpg"

        filepath = os.path.join(
            save_folder,
            filename
        )

        cv2.imwrite(
            filepath,
            hand_image
        )

        print(
            f"Saved: {filename} "
            f"({count}/{MAX_IMAGES})"
        )

    # Q
    elif key == ord("q"):

        print("\nCollection stopped.")

        break

# =========================
# CLEANUP
# =========================

camera.release()
hands.close()
cv2.destroyAllWindows()

print(f"\nFinished collecting {sign}.")
print(f"Total images: {count}")