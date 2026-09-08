import cv2
import threading

camera = None
camera_active = False

latest_sign = "No hand detected"

lock = threading.Lock()


def _open_capture():
    """Open the webcam using the fastest backend available on this OS."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback for non-Windows systems or if DirectShow isn't available
        cap = cv2.VideoCapture(0)

    # Smaller buffer = less lag reading the latest frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def start_camera():
    """Turn streaming on. Only opens the physical device the first time -
    after that it just resumes reading, so this is near-instant."""
    global camera, camera_active

    with lock:
        if camera is None or not camera.isOpened():
            camera = _open_capture()
        camera_active = True


def stop_camera():
    """Turn streaming off WITHOUT releasing the webcam hardware, so the
    next start_camera() call is instant instead of re-initializing."""
    global camera_active

    with lock:
        camera_active = False


def release_camera():
    """Fully release the webcam hardware. Call this only when the app is
    closing, not on every Close Camera click."""
    global camera, camera_active

    with lock:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None


def detect_hand_sign(frame):
    """
    Put your hand-sign detection model here.

    Example:
        if detected_sign == "thumbs_up":
            return "Good / Like"

    For now this is only a placeholder.
    """

    # TODO: Add your hand sign detection model here

    return "No hand detected"


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

        # Flip camera so it behaves like a mirror
        frame = cv2.flip(frame, 1)

        # Detect hand sign
        sign = detect_hand_sign(frame)

        # Store latest prediction
        with lock:
            latest_sign = sign

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # Send frame to browser
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


def get_latest_sign():
    with lock:
        return latest_sign