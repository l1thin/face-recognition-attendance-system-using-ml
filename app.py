from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
dataset_path = os.path.join(BASE_DIR, "dataset")
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
model_path = os.path.join(BASE_DIR, "trainer.yml")

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

attendance = []
today = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today}.csv"

# ---------------- TRAIN MODEL ----------------
def train_model():
    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for folder in os.listdir(dataset_path):
        path = os.path.join(dataset_path, folder)
        if not os.path.isdir(path):
            continue

        label_map[current_id] = folder

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            detected = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in detected:
                faces.append(img[y:y+h, x:x+w])
                labels.append(current_id)

        current_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        recognizer.save(model_path)

    return label_map


label_map = train_model()

# ---------------- VIDEO STREAM ----------------
def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70 and id_ in label_map:
                name = label_map[id_]
                time_now = datetime.now().strftime("%H:%M:%S")

                if name not in [row[0] for row in attendance]:
                    attendance.append([name, today, time_now])

                    df = pd.DataFrame(attendance, columns=["Name", "Date", "Time"])

                    if os.path.exists(attendance_file):
                        old_df = pd.read_csv(attendance_file)
                        df = pd.concat([old_df, df]).drop_duplicates(subset=["Name"], keep="first")

                    df.to_csv(attendance_file, index=False)

                text = name
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_data')
def attendance_data():
    return jsonify(attendance)


@app.route('/capture', methods=['POST'])
def capture():
    global label_map

    name = request.form['name'].strip().replace(" ", "_")
    student_path = os.path.join(dataset_path, name)

    if not os.path.exists(student_path):
        os.makedirs(student_path)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < 15:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(student_path, f"{count}.jpg"), face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Capturing {count}/15", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Registering Student - Press ESC to stop", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    label_map = train_model()

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)