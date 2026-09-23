import cv2
import threading
import joblib
import mediapipe as mp
import os
import numpy as np

from skimage.feature import hog


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "asl_model_mediapipe.pkl"

model = None
classes = []
IMG_SIZE = (64, 64)
PADDING = 50


if os.path.exists(MODEL_PATH):

    try:

        model_data = joblib.load(MODEL_PATH)

        model = model_data["model"]

        classes = model_data["classes"]

        IMG_SIZE = tuple(
            model_data.get(
                "img_size",
                (64, 64)
            )
        )

        PADDING = model_data.get(
            "padding",
            50
        )

        print("===================================")
        print("MediaPipe model loaded successfully")
        print("===================================")

        print("Model:", MODEL_PATH)
        print("Image size:", IMG_SIZE)
        print("Padding:", PADDING)
        print("Classes:", len(classes))

    except Exception as e:

        print("Could not load model.")
        print("Error:", e)

else:

    print("ERROR: Model not found:")
    print(MODEL_PATH)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

camera = None
camera_active = False

latest_sign = "No hand detected"

lock = threading.Lock()


def _open_capture():

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    return cap


def start_camera():

    global camera
    global camera_active

    with lock:

        if camera is None or not camera.isOpened():

            camera = _open_capture()

        camera_active = True


def stop_camera():

    global camera_active

    with lock:

        camera_active = False


def release_camera():

    global camera
    global camera_active

    with lock:

        camera_active = False

        if camera is not None:

            camera.release()

            camera = None


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
# HAND CROP
# ============================================================

def get_hand_data(frame):

    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb_frame
    )

    if not results.multi_hand_landmarks:

        return None, None


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

        return None, None


    roi = frame[
        y_min:y_max,
        x_min:x_max
    ]


    if roi.size == 0:

        return None, None


    return (
        (x_min, y_min, x_max, y_max),
        roi
    )


# ============================================================
# PREDICTION
# ============================================================

def detect_hand_sign(frame):

    hand_box, roi = get_hand_data(
        frame
    )


    if hand_box is None:

        return (
            "No hand detected",
            0.0,
            None
        )


    if model is None:

        return (
            "Model not loaded",
            0.0,
            hand_box
        )


    features = extract_hog(
        roi
    )


    prediction_index = model.predict(
        [features]
    )[0]


    prediction = classes[
        int(prediction_index)
    ]


    # ========================================================
    # CONFIDENCE-LIKE SCORE
    # ========================================================

    confidence = 0.0


    try:

        scores = model.decision_function(
            [features]
        )

        if scores.ndim == 2:

            scores = scores[0]


        exp_scores = np.exp(
            scores - np.max(scores)
        )

        probabilities = (
            exp_scores /
            np.sum(exp_scores)
        )


        confidence = (
            float(np.max(probabilities))
            * 100
        )


    except Exception:

        confidence = 0.0


    return (
        prediction,
        confidence,
        hand_box
    )


# ============================================================
# VIDEO STREAM
# ============================================================

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


        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )


        sign, confidence, hand_box = (
            detect_hand_sign(frame)
        )


        with lock:

            latest_sign = sign


        # ====================================================
        # DRAW HAND BOX
        # ====================================================

        if hand_box is not None:

            x1, y1, x2, y2 = hand_box


            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            label = (
                f"{sign}  "
                f"{confidence:.1f}%"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(30, y1 - 10)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


        else:

            cv2.putText(

                frame,

                "No hand detected",

                (30, 50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 255),

                2
            )


        # ====================================================
        # JPEG
        # ====================================================

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


# ============================================================
# LATEST SIGN
# ============================================================

def get_latest_sign():

    with lock:

        return latest_sign