# 🎯 Face Recognition Attendance System

A real-time Face Recognition based Attendance System built using Flask, OpenCV (LBPH), and Python.

This system allows:
- 📸 Live face recognition through webcam
- ➕ Registering new students directly from dashboard
- 🧠 Automatic model retraining
- 📝 Automatic attendance saving (date-wise CSV)
- 📊 Live attendance display on web interface


## 🧠 How It Works

1. Faces are detected using Haar Cascade.
2. Faces are recognized using LBPH (Local Binary Pattern Histogram).
3. When a face is recognized:
   - Name is displayed.
   - Attendance is automatically recorded.
   - Saved in a CSV file with date and time.
4. New students can be registered from the same dashboard.
5. After registration, the model retrains automatically.


## 📂 Project Structure

face_reco/
│
├── app.py
├── trainer.yml
├── haarcascade_frontalface_default.xml
├── attendance_YYYY-MM-DD.csv
│
├── dataset/
│   ├── Student_Name/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md


## ⚙️ Requirements

- Python 3.10 or 3.11 (Recommended)
- Webcam
- pip


## 📦 Installation & Setup Guide

1️⃣ Clone the Repository

git clone https://github.com/your-username/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System


2️⃣ Create Virtual Environment (Recommended)

python -m venv venv

Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate


3️⃣ Install Dependencies

pip install flask opencv-contrib-python numpy pandas


4️⃣ Run the Application

python app.py


5️⃣ Open in Browser

http://127.0.0.1:5000


## ➕ Register New Student

1. Enter student name in input field.
2. Click "Register Student".
3. Webcam opens.
4. 15 images are captured automatically.
5. Model retrains automatically.
6. Student is now ready for recognition.


## 📸 Mark Attendance

- When a registered student appears:
  - Name is detected.
  - Attendance is automatically recorded.
  - Stored in attendance_YYYY-MM-DD.csv.

No manual saving required.


## 🛠 Technologies Used

- Python
- Flask
- OpenCV
- LBPH Face Recognizer
- NumPy
- Pandas
- HTML / CSS


## 🚀 Features

- Real-time face recognition
- Automatic attendance system
- Duplicate attendance prevention
- Dashboard-based student registration
- Automatic model retraining
- Date-wise attendance storage
- Clean web interface


## ⚠️ Notes

- Make sure your webcam is connected.
- Use Python 3.10 or 3.11 for best compatibility.
- Virtual environment is recommended.


## ⭐ Future Improvements

- Database integration (MySQL / PostgreSQL)
- Login authentication system
- Attendance analytics
- Cloud deployment