# add_faces_deepface.py
import cv2
import pickle
import numpy as np
import os
import sys
from deepface import DeepFace

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_NAME = "Facenet"            # choose and keep same in test.py
DETECTOR_BACKEND = 'opencv'       # avoids extra deps

# build model first (downloads if needed)
try:
    print("Loading DeepFace model:", MODEL_NAME)
    model = DeepFace.build_model(MODEL_NAME)
    print("Model loaded.")
except Exception as e:
    print("ERROR building model:", e)
    print("Ensure you have internet on first run and installed dependencies: pip install deepface tensorflow")
    sys.exit(1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml'))

embeddings = []
i = 0
name = input("Enter your name: ").strip()
MAX_EMB = 50   # embeddings per person

print("Start capturing. Press 'q' to stop earlier.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        # ignore tiny faces
        if w < 50 or h < 50:
            continue

        if i % 10 == 0 and len(embeddings) < MAX_EMB:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                rep = DeepFace.represent(rgb, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
                # extract embedding robustly
                if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], dict) and 'embedding' in rep[0]:
                    emb = np.array(rep[0]['embedding']).flatten()
                elif isinstance(rep, dict) and 'embedding' in rep:
                    emb = np.array(rep['embedding']).flatten()
                else:
                    emb = np.array(rep).flatten()
                embeddings.append(emb)
                print(f"Collected embeddings: {len(embeddings)} / {MAX_EMB}")
            except Exception as e:
                print("Embedding error (skipping frame):", e)
        i += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,255), 1)

    cv2.putText(frame, f"Embeddings: {len(embeddings)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Collect", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(embeddings) >= MAX_EMB:
        break

video.release()
cv2.destroyAllWindows()

if len(embeddings) == 0:
    print("No embeddings collected. Try better light / larger face.")
    sys.exit(1)

embeddings = np.asarray(embeddings)
print("Collected embeddings shape:", embeddings.shape)

# save labels and embeddings (keeps counts consistent)
names_path = os.path.join(DATA_DIR, 'names.pkl')
faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')

if not os.path.exists(names_path):
    names = [name] * len(embeddings)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * len(embeddings)

with open(names_path, 'wb') as f:
    pickle.dump(names, f)

if not os.path.exists(faces_path):
    saved_faces = embeddings
else:
    with open(faces_path, 'rb') as f:
        saved_faces = pickle.load(f)
    saved_faces = np.vstack([saved_faces, embeddings])

with open(faces_path, 'wb') as f:
    pickle.dump(saved_faces, f)

print(f"Saved {len(embeddings)} embeddings for {name}")
