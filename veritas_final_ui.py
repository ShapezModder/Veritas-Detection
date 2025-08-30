# veritas_final_ui.py
import os
import sys
import cv2
import numpy as np
from datetime import datetime
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter.scrolledtext import ScrolledText
from PIL import Image
import face_recognition
from ultralytics import YOLO
import pygame

# ---------------- Path Helper ----------------
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller exe"""
    try:
        base_path = sys._MEIPASS  # PyInstaller extracts to a temp folder
    except Exception:
        base_path = os.path.abspath(".")  # Running as script
    return os.path.join(base_path, relative_path)

# ---------------- Config ----------------
WINDOW_W, WINDOW_H = 1280, 720
LEFT_W, LEFT_H = 960, 540
LEFT_X, LEFT_Y = 20, 100
RIGHT_X, RIGHT_Y = 1000, 100

LOG_W, LOG_H = 320, LEFT_H
COMBINED_SIZE = (LEFT_W + LOG_W, LEFT_H)

FPS = 20.0
CODEC = "XVID"

#  All resource paths go through resource_path
BG_PATH = resource_path("background.png")
SPLASH_PATH = resource_path("splash.png")
YOLO_WEIGHTS = resource_path("yolov8n.pt")
ALARM_SOUND = resource_path("siren.wav")

MOTION_DIR = "MOTION"
HUMAN_DIR = "HUMAN"
FACE_DIR = "FACE"

MOTION_MIN_AREA = 1200
NO_MOTION_STOP_FRAMES = 20
NO_HUMAN_STOP_FRAMES = 20
NO_FACE_STOP_FRAMES = 20

# ---------------- Helpers ----------------
def ensure_dirs():
    for d in (MOTION_DIR, HUMAN_DIR, FACE_DIR):
        os.makedirs(d, exist_ok=True)

def timestamp_filename(prefix, folder):
    ts = datetime.now().strftime("%d_%m_%Y  TIME %H_%M_%S")
    return os.path.join(folder, f"{prefix}_{ts}.avi")

def draw_log_panel(messages, width=LOG_W, height=LOG_H):
    panel = np.full((height, width, 3), 18, dtype=np.uint8)  # dark panel
    y = 28
    for msg in messages[-18:]:
        cv2.putText(panel, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 250, 190), 1, cv2.LINE_AA)
        y += 26
        if y > height - 10:
            break
    cv2.rectangle(panel, (0, 0), (width-1, height-1), (50, 60, 70), 1)
    return panel

# ---------------- App ----------------
class VeritasApp:
    def __init__(self):
        # CTk setup
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("Veritas Detection v2.0")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.resizable(False, False)
        self._center_window()

        ensure_dirs()

        # models + camera
        self.yolo = YOLO(YOLO_WEIGHTS)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LEFT_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LEFT_H)

        # pygame mixer for alarm
        try:
            pygame.mixer.init()
        except Exception:
            pass
        self.alarm_playing = False

        # state
        self.motion_on = ctk.BooleanVar(value=False)
        self.human_on = ctk.BooleanVar(value=False)
        self.face_on = ctk.BooleanVar(value=False)
        self.alarm_on = ctk.BooleanVar(value=False)

        # recording writers
        self.writer_motion = None
        self.writer_human = None
        self.writer_face = None

        # counters
        self.nomotion_frames = 0
        self.nohuman_frames = 0
        self.noface_frames = 0

        self.prev_gray = None
        self.log_buffer = []

        self._init_ui()

        self.root.after(1200, self.loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def _center_window(self):
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w // 2) - (WINDOW_W // 2)
        y = (screen_h // 2) - (WINDOW_H // 2)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")

    def _init_ui(self):
        # Background
        if os.path.exists(BG_PATH):
            bg_pil = Image.open(BG_PATH).resize((WINDOW_W, WINDOW_H))
            bg_img = CTkImage(light_image=bg_pil, dark_image=bg_pil,
                              size=(WINDOW_W, WINDOW_H))
            self.bg_label = ctk.CTkLabel(self.root, image=bg_img, text="")
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.root.configure(fg_color="#0b0f14")

        # Video area
        self.video_label = ctk.CTkLabel(self.root, text="", width=LEFT_W,
                                        height=LEFT_H, fg_color="black")
        self.video_label.place(x=LEFT_X, y=LEFT_Y)

        # Splash
        if os.path.exists(SPLASH_PATH):
            splash_pil = Image.open(SPLASH_PATH).resize((LEFT_W, LEFT_H))
            splash_img = CTkImage(light_image=splash_pil, dark_image=splash_pil,
                                  size=(LEFT_W, LEFT_H))
            self.video_label.configure(image=splash_img)
            self.video_label.image = splash_img

        # Controls
        controls_w, controls_h = 250, 200
        self.controls_frame = ctk.CTkFrame(self.root, corner_radius=0,
                                           fg_color="#131516", border_width=0,
                                           width=controls_w, height=controls_h)
        self.controls_frame.place(x=RIGHT_X, y=RIGHT_Y)

        lbl = ctk.CTkLabel(self.controls_frame, text="Controls",
                           font=("Segoe UI", 15, "bold"), text_color="#EAEAEA")
        lbl.pack(anchor="w", padx=12, pady=(10, 6))

        switch_font = ("Segoe UI", 13)
        self.sw_motion = ctk.CTkSwitch(self.controls_frame, text="Motion Detection",
                                       variable=self.motion_on,
                                       command=self._on_switch_change,
                                       progress_color="#00D084",
                                       button_color="#F6F6F6",
                                       text_color="#FFFFFF", font=switch_font)
        self.sw_motion.pack(anchor="w", padx=12, pady=6)

        self.sw_human = ctk.CTkSwitch(self.controls_frame, text="Human Detection",
                                      variable=self.human_on,
                                      command=self._on_switch_change,
                                      progress_color="#00D084",
                                      button_color="#F6F6F6",
                                      text_color="#FFFFFF", font=switch_font)
        self.sw_human.pack(anchor="w", padx=12, pady=6)

        self.sw_face = ctk.CTkSwitch(self.controls_frame, text="Face Detection",
                                     variable=self.face_on,
                                     command=self._on_switch_change,
                                     progress_color="#00D084",
                                     button_color="#F6F6F6",
                                     text_color="#FFFFFF", font=switch_font)
        self.sw_face.pack(anchor="w", padx=12, pady=6)

        self.sw_alarm = ctk.CTkSwitch(self.controls_frame, text="Alarm (siren on human)",
                                      variable=self.alarm_on,
                                      command=self._on_switch_change,
                                      progress_color="#00D084",
                                      button_color="#F6F6F6",
                                      text_color="#FFFFFF", font=switch_font)
        self.sw_alarm.pack(anchor="w", padx=12, pady=6)

        # Console
        console_w, console_h = 250, 280
        self.console_frame = ctk.CTkFrame(self.root, corner_radius=0,
                                          fg_color="#131516", border_width=0,
                                          width=console_w, height=console_h)
        self.console_frame.place(x=RIGHT_X, y=RIGHT_Y + controls_h + 20)

        lblc = ctk.CTkLabel(self.console_frame, text="Console",
                            font=("Segoe UI", 15, "bold"), text_color="#EAEAEA")
        lblc.pack(anchor="w", padx=12, pady=(10, 6))

        self.console = ScrolledText(self.console_frame, width=30, height=12,
                                    bg="#061012", fg="#7df089",
                                    insertbackground="#7df089",
                                    font=("Consolas", 11), relief="flat")
        self.console.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.console.configure(state="disabled")

        self._update_switch_text_colors()

    def _on_switch_change(self):
        self._update_switch_text_colors()
        s = f"Motion={'ON' if self.motion_on.get() else 'OFF'} | " \
            f"Human={'ON' if self.human_on.get() else 'OFF'} | " \
            f"Face={'ON' if self.face_on.get() else 'OFF'} | " \
            f"Alarm={'ON' if self.alarm_on.get() else 'OFF'}"
        self.append_log(s)

    def _update_switch_text_colors(self):
        for sw in (self.sw_motion, self.sw_human, self.sw_face, self.sw_alarm):
            try:
                sw.configure(text_color="#FFFFFF")
            except Exception:
                pass

    # ---------- main loop ----------
    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            self.append_log("Camera read failed.")
            self.root.after(50, self.loop)
            return

        frame = cv2.resize(frame, (LEFT_W, LEFT_H))
        display_frame = frame.copy()

        # Face detection
        small_rgb = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25),
                                 cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(small_rgb)
        faces_scaled = []
        for (t, r, b, l) in faces:
            t, r, b, l = t*4, r*4, b*4, l*4
            faces_scaled.append((t, r, b, l))
            cv2.rectangle(display_frame, (l, t), (r, b), (0, 255, 0), 2)

        # YOLO detection
        persons_detected = False
        results = self.yolo(display_frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{self.yolo.names[cls]} {conf:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, label, (x1, max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                if self.yolo.names[cls].lower() == "person":
                    persons_detected = True

        # Motion detection
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21,21), 0)
        motion_detected = False
        if self.prev_gray is None:
            self.prev_gray = gray
        else:
            frame_delta = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < MOTION_MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(display_frame, (x,y), (x+w, y+h), (0,255,255), 2)
                motion_detected = True
            self.prev_gray = gray

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

        # Logs
        if motion_detected:
            self.append_log(f"[{ts}] Motion detected")
        if persons_detected:
            self.append_log(f"[{ts}] Human detected")
        if faces_scaled:
            self.append_log(f"[{ts}] Face detected ({len(faces_scaled)})")

        # Alarm
        if self.alarm_on.get() and persons_detected:
            if not self.alarm_playing:
                try:
                    if os.path.exists(ALARM_SOUND):
                        pygame.mixer.music.load(ALARM_SOUND)
                        pygame.mixer.music.play(-1)
                        self.alarm_playing = True
                except Exception:
                    pass
        else:
            if self.alarm_playing:
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
                self.alarm_playing = False

        # Recording
        combined_for_save = self._build_combined_frame(display_frame)

        # Motion recording
        if self.motion_on.get():
            if motion_detected:
                if self.writer_motion is None:
                    path = timestamp_filename("Motion", MOTION_DIR)
                    self.writer_motion = cv2.VideoWriter(path,
                        cv2.VideoWriter_fourcc(*CODEC), FPS, COMBINED_SIZE)
                    self.append_log(f"Recording started: {path}")
                self.nomotion_frames = 0
                self.writer_motion.write(combined_for_save)
            else:
                if self.writer_motion is not None:
                    self.nomotion_frames += 1
                    if self.nomotion_frames >= NO_MOTION_STOP_FRAMES:
                        self.writer_motion.release()
                        self.writer_motion = None
                        self.append_log("Motion recording stopped.")
                        self.nomotion_frames = 0
        else:
            if self.writer_motion is not None:
                self.writer_motion.release()
                self.writer_motion = None
                self.append_log("Motion recording stopped (toggle OFF).")

        # Human recording
        if self.human_on.get():
            if persons_detected:
                if self.writer_human is None:
                    path = timestamp_filename("Human", HUMAN_DIR)
                    self.writer_human = cv2.VideoWriter(path,
                        cv2.VideoWriter_fourcc(*CODEC), FPS, COMBINED_SIZE)
                    self.append_log(f"Recording started: {path}")
                self.nohuman_frames = 0
                self.writer_human.write(combined_for_save)
            else:
                if self.writer_human is not None:
                    self.nohuman_frames += 1
                    if self.nohuman_frames >= NO_HUMAN_STOP_FRAMES:
                        self.writer_human.release()
                        self.writer_human = None
                        self.append_log("Human recording stopped.")
                        self.nohuman_frames = 0
        else:
            if self.writer_human is not None:
                self.writer_human.release()
                self.writer_human = None
                self.append_log("Human recording stopped (toggle OFF).")

        # Face recording
        if self.face_on.get():
            if faces_scaled:
                if self.writer_face is None:
                    path = timestamp_filename("Face", FACE_DIR)
                    self.writer_face = cv2.VideoWriter(path,
                        cv2.VideoWriter_fourcc(*CODEC), FPS, COMBINED_SIZE)
                    self.append_log(f"Recording started: {path}")
                self.noface_frames = 0
                self.writer_face.write(combined_for_save)
            else:
                if self.writer_face is not None:
                    self.noface_frames += 1
                    if self.noface_frames >= NO_FACE_STOP_FRAMES:
                        self.writer_face.release()
                        self.writer_face = None
                        self.append_log("Face recording stopped.")
                        self.noface_frames = 0
        else:
            if self.writer_face is not None:
                self.writer_face.release()
                self.writer_face = None
                self.append_log("Face recording stopped (toggle OFF).")

        # Render
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        ctk_img = CTkImage(light_image=pil, dark_image=pil,
                           size=(LEFT_W, LEFT_H))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

        self.root.after(int(1000 / FPS), self.loop)

    def _build_combined_frame(self, display_frame):
        drawn_log = draw_log_panel(self.log_buffer, LOG_W, LOG_H)
        combined = cv2.hconcat([display_frame, drawn_log])
        return cv2.resize(combined, COMBINED_SIZE)

    def append_log(self, line):
        self.console.configure(state="normal")
        self.console.insert("end", line + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")
        self.log_buffer.append(line if len(line) < 95 else line[:92] + "…")
        self.log_buffer = self.log_buffer[-200:]

    def on_close(self):
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if self.writer_motion is not None:
                self.writer_motion.release()
            if self.writer_human is not None:
                self.writer_human.release()
            if self.writer_face is not None:
                self.writer_face.release()
        except Exception:
            pass
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.root.destroy()

# ---------------- Run ----------------
if __name__ == "__main__":
    VeritasApp()
