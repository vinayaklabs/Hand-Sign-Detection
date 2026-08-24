"""
SignSpeak - Step 1: Camera permission flow (Python / Tkinter / OpenCV)

Screens:
  1. Ask permission        -> Yes / Not now
  2a. Ready (granted)      -> "Open camera" opens the feed directly
  2b. Blocked (denied)     -> "Open camera" tries again (re-asks / retries access)
  3. Live camera feed

Install first:
    pip install opencv-python pillow

Run:
    python camera_app.py
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk

# ---- Basic, simple color palette ----
BG = "#F4F4F5"
CARD = "#FFFFFF"
BORDER = "#D9D9DE"
TEXT = "#1F2328"
TEXT_DIM = "#5B616E"
BLUE = "#2563EB"
BLUE_HOVER = "#1D4ED8"
RED = "#D64545"


class SignSpeakApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignSpeak — Camera Setup")
        self.root.geometry("420x520")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.title_font = tkfont.Font(family="Arial", size=18, weight="bold")
        self.status_font = tkfont.Font(family="Arial", size=10, weight="bold")
        self.body_font = tkfont.Font(family="Arial", size=11)
        self.btn_font = tkfont.Font(family="Arial", size=11, weight="bold")

        self.cap = None          # cv2.VideoCapture
        self.video_label = None
        self.streaming = False

        self.card = tk.Frame(self.root, bg=CARD, highlightbackground=BORDER,
                              highlightthickness=1, bd=0)
        self.card.place(x=20, y=20, width=380, height=480)

        self.show_ask_screen()

    # ---------- helpers ----------
    def clear_card(self):
        for widget in self.card.winfo_children():
            widget.destroy()

    def make_button(self, parent, text, command, primary=True, danger=False):
        bg = BLUE if primary else (RED if danger else CARD)
        fg = "#FFFFFF" if primary else (RED if danger else TEXT)
        border = BLUE if primary else (RED if danger else BORDER)
        btn = tk.Button(
            parent, text=text, command=command, font=self.btn_font,
            bg=bg, fg=fg, activebackground=BLUE_HOVER if primary else bg,
            activeforeground=fg, relief="flat", bd=0,
            highlightbackground=border, highlightthickness=1,
            padx=14, pady=10, cursor="hand2"
        )
        return btn

    def status_label(self, parent, text):
        return tk.Label(parent, text=text.upper(), font=self.status_font,
                bg=CARD, fg=TEXT_DIM, anchor="w")

    # ---------- Screen 1: ask permission ----------
    def show_ask_screen(self):
        self.stop_camera()
        self.clear_card()
        pad = {"padx": 24}

        self.status_label(self.card, "Permission required").pack(anchor="w", pady=(24, 10), **pad)
        tk.Label(self.card, text="SignSpeak needs your camera", font=self.title_font,
                 bg=CARD, fg=TEXT, wraplength=330, justify="left").pack(anchor="w", **pad)
        tk.Label(self.card, text=("We use your camera to read hand signs live and turn them "
                                   "into text and voice. Nothing is recorded or sent anywhere."),
                 font=self.body_font, bg=CARD, fg=TEXT_DIM, wraplength=330,
                 justify="left").pack(anchor="w", pady=(10, 24), **pad)

        row = tk.Frame(self.card, bg=CARD)
        row.pack(fill="x", **pad)
        self.make_button(row, "Not now", self.show_blocked_screen, primary=False).pack(
            side="left", expand=True, fill="x", padx=(0, 6))
        self.make_button(row, "Allow camera", self.show_ready_screen, primary=True).pack(
            side="left", expand=True, fill="x", padx=(6, 0))

    # ---------- Screen 2a: granted, ready to open ----------
    def show_ready_screen(self):
        self.clear_card()
        pad = {"padx": 24}

        self.status_label(self.card, "Camera access granted").pack(anchor="w", pady=(24, 10), **pad)
        tk.Label(self.card, text="You're all set", font=self.title_font,
                 bg=CARD, fg=TEXT).pack(anchor="w", **pad)
        tk.Label(self.card, text="Open the camera whenever you're ready to start signing.",
                 font=self.body_font, bg=CARD, fg=TEXT_DIM, wraplength=330,
                 justify="left").pack(anchor="w", pady=(10, 24), **pad)

        self.make_button(self.card, "Open camera", self.open_camera, primary=True).pack(
            fill="x", **pad)

    # ---------- Screen 2b: denied, still offer to open (retries access) ----------
    def show_blocked_screen(self):
        self.clear_card()
        pad = {"padx": 24}

        self.status_label(self.card, "Camera access off").pack(anchor="w", pady=(24, 10), **pad)
        tk.Label(self.card, text="Camera is turned off", font=self.title_font,
                 bg=CARD, fg=TEXT).pack(anchor="w", **pad)
        tk.Label(self.card, text=("Sign detection won't work until camera access is allowed. "
                                   "Tap below whenever you're ready."),
                 font=self.body_font, bg=CARD, fg=TEXT_DIM, wraplength=330,
                 justify="left").pack(anchor="w", pady=(10, 24), **pad)

        self.make_button(self.card, "Open camera", self.open_camera, primary=True).pack(
            fill="x", **pad)
        tk.Label(self.card, text="This will try to access the camera again.",
                 font=self.body_font, bg=CARD, fg=TEXT_DIM).pack(pady=(10, 0))

    # ---------- Screen 3: live feed ----------
    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error_screen("Could not access the camera. It may be in use by "
                                    "another app, or not connected.")
            return
        self.show_live_screen()
        self.streaming = True
        self.update_frame()

    def show_live_screen(self):
        self.clear_card()
        pad = {"padx": 24}

        self.status_label(self.card, "Live").pack(anchor="w", pady=(20, 8), **pad)

        self.video_label = tk.Label(self.card, bg="#000000")
        self.video_label.pack(padx=24, pady=(0, 10))

        caption_box = tk.Frame(self.card, bg=BG, highlightbackground=BORDER, highlightthickness=1)
        caption_box.pack(fill="x", padx=24, pady=(0, 14))
        tk.Label(caption_box, text="DETECTED SIGN", font=self.status_font,
                 bg=BG, fg=TEXT_DIM).pack(anchor="w", padx=12, pady=(8, 0))
        self.caption_text = tk.Label(caption_box, text="Waiting for a hand sign…",
                                      font=self.title_font, bg=BG, fg=TEXT)
        self.caption_text.pack(anchor="w", padx=12, pady=(0, 8))

        self.make_button(self.card, "Stop camera", self.show_ready_screen,
                          primary=False, danger=True).pack(fill="x", padx=24)

    def update_frame(self):
        if not self.streaming or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (330, 250))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

            # Your hand-sign detector plugs in here:
            # sign = detector.predict(frame)
            # self.caption_text.config(text=sign)

        self.root.after(15, self.update_frame)

    def stop_camera(self):
        self.streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------- Screen 4: error ----------
    def show_error_screen(self, message):
        self.clear_card()
        pad = {"padx": 24}

        self.status_label(self.card, "Access blocked").pack(anchor="w", pady=(24, 10), **pad)
        tk.Label(self.card, text="Camera access is blocked", font=self.title_font,
                 bg=CARD, fg=TEXT, wraplength=330, justify="left").pack(anchor="w", **pad)
        tk.Label(self.card, text=message, font=self.body_font, bg=CARD, fg=TEXT_DIM,
                 wraplength=330, justify="left").pack(anchor="w", pady=(10, 24), **pad)

        self.make_button(self.card, "Try again", self.open_camera, primary=True).pack(
            fill="x", **pad)


if __name__ == "__main__":
    root = tk.Tk()
    app = SignSpeakApp(root)
    root.mainloop()