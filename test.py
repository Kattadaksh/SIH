# test_deepface_manual.py
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from deepface import DeepFace
import subprocess
import sys

# ----------------------------- CONFIG -----------------------------
DATA_DIR = 'data'
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = 'opencv'
HOTSPOT_SSID = "Teacher_Hotspot"
COL_NAMES = ['NAME', 'TIME']
IMG_BACKGROUND_PATH = ""

# ----------------------------- FUNCTIONS -------------------------
def speak(text):
    """Text-to-speech"""
    try:
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    except Exception as e:
        print("TTS error:", e)

def is_connected_to_wifi(target_ssid="MNNIT"):
    """Check if connected to specified Wi-Fi SSID"""
    try:
        output = subprocess.check_output("netsh wlan show interfaces", shell=True).decode()
        for line in output.splitlines():
            if "SSID" in line and "BSSID" not in line:
                current_ssid = line.split(":", 1)[1].strip()
                return current_ssid == target_ssid
        return False
    except Exception as e:
        print("Wi-Fi check error:", e)
        return False

# ----------------------------- LOAD MODEL ------------------------
try:
    print("Loading DeepFace model...")
    model = DeepFace.build_model(MODEL_NAME)
    print("Model ready.")
except Exception as e:
    print("Model build failed:", e)
    sys.exit(1)

# ----------------------------- LOAD DATA -------------------------
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml'))

with open(os.path.join(DATA_DIR, 'names.pkl'), 'rb') as f:
    LABELS = pickle.load(f)
with open(os.path.join(DATA_DIR, 'faces_data.pkl'), 'rb') as f:
    FACES = pickle.load(f)

# Ensure matching lengths
min_len = min(FACES.shape[0], len(LABELS))
FACES = FACES[:min_len]
LABELS = LABELS[:min_len]

# Embedding dimension
emb_dim = FACES.shape[1]

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(FACES, LABELS)
print("Classifier trained.")

# Load background image
imgBackground = cv2.imread(IMG_BACKGROUND_PATH)

# ----------------------------- ATTENDANCE -----------------------
marked = set()
attendance = None
filepath = None
exist = False

# ----------------------------- MAIN LOOP ------------------------
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w < 50 or h < 50:
            continue

        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # ------------------ FACE EMBEDDING ------------------
        try:
            rep = DeepFace.represent(rgb, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
            if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
                emb = np.array(rep[0]['embedding']).flatten()
            elif isinstance(rep, dict) and 'embedding' in rep:
                emb = np.array(rep['embedding']).flatten()
            else:
                emb = np.array(rep).flatten()
        except Exception as e:
            print("Embedding error (skipped):", e)
            continue

        if emb.shape[0] != emb_dim:
            print("Embedding size mismatch, skipping.")
            continue

        # ------------------ PREDICTION ------------------
        out = knn.predict(emb.reshape(1, -1))[0]
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        filepath = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(filepath)

        # ------------------ DISPLAY ------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(out), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        if out not in marked:
            attendance = [str(out), str(timestamp)]

    # Show frame
    try:
        if imgBackground is not None:
            imgBackground[162:162+480, 55:55+640] = frame
            cv2.imshow("Frame", imgBackground)
        else:
            cv2.imshow("Frame", frame)
    except:
        cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    # ------------------ MANUAL ATTENDANCE ------------------
    if k == ord('o') and attendance is not None:
        if not is_connected_to_wifi("Shobhit's A55"):
            speak("You are not connected to the hotspot. Attendance denied.")
            print("Not connected to hotspot.")
            attendance = None
        else:
            name = attendance[0]
            if name not in marked:
                speak(f"Press 'o' again to confirm attendance for {name}")
                print(f"Press 'o' again to confirm attendance for {name}")
                key = cv2.waitKey(0)
                if key == ord('o'):
                    marked.add(name)
                    speak(f"Attendance taken for {name}")
                    # Save attendance to CSV
                    if exist:
                        with open(filepath, "a", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(attendance)
                    else:
                        with open(filepath, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(COL_NAMES)
                            writer.writerow(attendance)
                    attendance = None

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
