from flask import Flask, send_file, Response, jsonify
import webbrowser
import threading

from Camera_app import generate_frames, get_latest_sign, start_camera, stop_camera

app = Flask(__name__)


@app.route("/")
def home():
    return send_file("index.html", mimetype="text/html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_sign")
def get_sign():
    return jsonify({"sign": get_latest_sign()})


@app.route("/start_camera", methods=["POST"])
def start_camera_route():
    start_camera()
    return jsonify({"status": "started"})


@app.route("/stop_camera", methods=["POST"])
def stop_camera_route():
    stop_camera()
    return jsonify({"status": "stopped"})


if __name__ == "__main__":

    print("SignSpeak is running...")
    print("Open: http://localhost:5000")

    threading.Timer(
        1,
        lambda: webbrowser.open("http://localhost:5000")
    ).start()

    app.run(
        host="localhost",
        port=5000,
        debug=False
    )