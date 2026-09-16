from flask import Flask, send_file, Response, jsonify, request , redirect , send_from_directory
import webbrowser
import threading
from database import create_user , login_user
from Camera_app import generate_frames, get_latest_sign, start_camera, stop_camera

app = Flask(__name__)


@app.route("/")
def home():
    return send_file("Pages/home.html", mimetype="text/html")

@app.route("/detect")
def detect():
    return send_file("index.html", mimetype="text/html")

@app.route("/about")
def about():
    return send_file("Pages/About.html", mimetype="text/html")

@app.route("/contact-us")
def contact_us():
    return send_file("Pages/Contact_Us.html", mimetype="text/html")

@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory("img", filename)

@app.route("/login")
def login():
    return send_file("Pages/login.html", mimetype="text/html")

@app.route("/sign-in")
def Sign_In_page():
    return send_file("Pages/Sign_In.html", mimetype="text/html")

@app.route("/sign-in" , methods=["POST"])
def sign_in():
    
    email = request.form["email"]
    password = request.form["password"]
    
    name = login_user(email , password)
    
    if(name):
        return redirect("/")
    
    else:
        return "Invalid Email or Password"

@app.route("/signup", methods=["POST"])
def signup():

    name = request.form["name"]
    email = request.form["email"]
    password = request.form["password"]

    success = create_user(name, email, password)

    if success:
        return redirect("/")

    else:
        return "Email already exists!"

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