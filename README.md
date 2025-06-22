# Emotion-X Web Platform

**Emotion-X** is a web-based emotion detection system that allows users to register/login using a Flask-powered backend and interact with a real-time AI model that detects human emotions based on facial gestures.

## 📦 About the Project

This project includes:

- 🧠 An AI model that detects real-time facial and gesture-based emotions using OpenCV, Mediapipe, and Streamlit.
- 🔐 A Flask-based user authentication system (login/register/logout).
- 🖥️ A web dashboard that links to the deployed AI model.

---

## ⚠️ Note on AI Model Deployment

Due to **Render's free tier memory limitations**, the AI model may sometimes **fail to load** or **timeout**.

If the AI model fails to open or stops working:

👉 **Step 1**: Try launching the AI model directly by clicking this link:  
🔗 [Launch AI Model] [LINK](https://emotion-x-app-19.onrender.com)


👉 **Step 2**: Once the AI model loads fully, return to the web app link below and use it normally.

---

## 🌐 Web App (Flask Project)

To check out the full web app with login, registration, and dashboard:

🔗 [Visit the Emotion-X Web App][LINK](https://your-flask-app.onrender.com)


_(Replace this with your actual Flask app Render link.)_

---

## 🛠️ Tech Stack

- Python 3.9+
- Flask
- Streamlit
- OpenCV
- Mediapipe
- JSON (for user data)
- HTML/CSS (for templates)

---

## 💡 Features

- Real-time emotion detection using webcam
- User authentication (Register/Login)
- AI model launched via dashboard
- Deployment-ready with Render

---

## 📂 Project Structure
Emotion-X project/
├── Emotion-X app/               # Flask app
│   ├── app.py
│   ├── templates/
│   │   ├── login.html
│   │   ├── register.html
│   │   └── dashboard.html
│   └── users.json
├── face_emotion_app/           # AI model code
│   └── emotion_x.py
├── requirements.txt
├── runtime.txt
└── README.md


