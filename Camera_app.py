import cv2
import threading
import joblib
import mediapipe as mp

from skimage.feature import hog


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "asl_model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
classes = model_data["classes"]
IMG_SIZE = tuple(model_data["img_size"])


# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
# CAMERA VARIABLES
# ==========================================

camera = None
camera_active = False

latest_sign = "No hand detected"

lock = threading.Lock()


# ==========================================
# OPEN CAMERA
# ==========================================

def _open_capture():
    """Open the webcam."""

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ==========================================
# START CAMERA
# ==========================================

def start_camera():
    global camera, camera_active

    with lock:

        if camera is None or not camera.isOpened():
            camera = _open_capture()

        camera_active = True


# ==========================================
# STOP CAMERA
# ==========================================

def stop_camera():
    global camera_active

    with lock:
        camera_active = False


# ==========================================
# RELEASE CAMERA
# ==========================================

def release_camera():
    global camera, camera_active

    with lock:

        camera_active = False

        if camera is not None:
            camera.release()
            camera = None


# ==========================================
# EXTRACT HOG FEATURES
# ==========================================

def extract_hog(image):

    image = cv2.resize(image, IMG_SIZE)

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


# ==========================================
# DETECT HAND
# ==========================================

def get_hand_box(frame):

    height, width = frame.shape[:2]

    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    x_coordinates = []
    y_coordinates = []

    for landmark in hand_landmarks.landmark:

        x_coordinates.append(
            int(landmark.x * width)
        )

        y_coordinates.append(
            int(landmark.y * height)
        )

    # Find hand boundaries
    x_min = max(0, min(x_coordinates))
    y_min = max(0, min(y_coordinates))

    x_max = min(width, max(x_coordinates))
    y_max = min(height, max(y_coordinates))

    # Add padding around hand
    padding = 30

    x1 = max(0, x_min - padding)
    y1 = max(0, y_min - padding)

    x2 = min(width, x_max + padding)
    y2 = min(height, y_max + padding)

    # Make sure the box is not too small
    if (x2 - x1) < 40 or (y2 - y1) < 40:
        return None

    return x1, y1, x2, y2


# ==========================================
# DETECT HAND SIGN
# ==========================================

def detect_hand_sign(frame):

    hand_box = get_hand_box(frame)

    # No hand
    if hand_box is None:
        return "No hand detected", 0.0, None

    x1, y1, x2, y2 = hand_box

    # Crop hand
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "No hand detected", 0.0, None

    # Extract HOG features
    features = extract_hog(roi)

    # Prediction
    prediction = model.predict([features])[0]

    # --------------------------------------
    # Confidence-like score
    # --------------------------------------

    decision_scores = model.decision_function([features])

    if decision_scores.ndim == 2:

        scores = decision_scores[0]

        # Convert decision values into a
        # confidence-like percentage
        exp_scores = __import__("numpy").exp(
            scores - scores.max()
        )

        probabilities = exp_scores / exp_scores.sum()

        confidence = probabilities.max() * 100

    else:

        confidence = 0.0

    return prediction, confidence, hand_box


# ==========================================
# GENERATE VIDEO FRAMES
# ==========================================

def generate_frames():

    global latest_sign

    while True:

        with lock:

            active = camera_active
            cam = camera

        if not active or cam is None:
            break

        success, frame = cam.read()

        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # ==================================
        # DETECT SIGN
        # ==================================

        sign, confidence, hand_box = detect_hand_sign(frame)

        with lock:
            latest_sign = sign

        # ==================================
        # DRAW HAND BOX
        # ==================================

        if hand_box is not None:

            x1, y1, x2, y2 = hand_box

            # Green box around actual hand
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Text shown above hand
            label = f"{sign}  {confidence:.1f}%"

            text_y = max(30, y1 - 10)

            # Black background behind text
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )[0]

            cv2.rectangle(
                frame,
                (x1, text_y - text_size[1] - 10),
                (x1 + text_size[0] + 10, text_y + 5),
                (0, 0, 0),
                -1
            )

            # Prediction text
            cv2.putText(
                frame,
                label,
                (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            # No hand message
            cv2.putText(
                frame,
                "No hand detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # ==================================
        # ENCODE FRAME
        # ==================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==========================================
# GET LATEST SIGN
# ==========================================

def get_latest_sign():

    with lock:
        return latest_sign
