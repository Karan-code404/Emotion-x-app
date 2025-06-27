import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.title("Real-Time Gesture Based Emotion Detection")

# Webcam access checkbox and start button
use_webcam = st.checkbox("Allow Webcam Access")
start_button = st.button("GO")

def get_distance(p1, p2):
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

frame_placeholder = st.empty()

if use_webcam and start_button:
    cap = cv2.VideoCapture(0)
    with mp_face.FaceMesh(max_num_faces=1) as face_mesh, \
         mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            emotion = "Neutral"

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                points = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    points.append((x, y))

                # ========== EYEBROW DOTS (GREEN) ==========
                cv2.circle(frame, points[70], 5, (0, 255, 0), -1)    # Left eyebrow - far left
                cv2.circle(frame, points[52], 5, (0, 255, 0), -1)    # Left eyebrow - middle
                cv2.circle(frame, points[282], 5, (0, 255, 0), -1)   # Right eyebrow - middle
                cv2.circle(frame, points[300], 5, (0, 255, 0), -1)   # Right eyebrow - far right

                # Angry detection based on eyebrow position
                left_brow_left = points[70][1]
                left_brow_right = points[52][1]
                right_brow_left = points[282][1]
                right_brow_right = points[300][1]

                if right_brow_right < right_brow_left and left_brow_left < left_brow_right:
                    emotion = "😠 Angry (Eyebrow Shape)"

                # Smile and laugh detection
                left_mouth = points[61]
                right_mouth = points[291]
                top_lip = points[13]
                bottom_lip = points[14]

                mouth_width = get_distance(left_mouth, right_mouth)
                mouth_height = get_distance(top_lip, bottom_lip)

                if mouth_width > 60 and mouth_height < 20:
                    emotion = "😊 Happy (Smiling)"
                if mouth_width > 70 and mouth_height > 12:
                    emotion = "😂 Laughing (Teeth showing)"

            if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    hand_tip = hand_landmarks.landmark[8]
                    hx, hy = int(hand_tip.x * img_w), int(hand_tip.y * img_h)

                    forehead = points[10]
                    left_eye = points[159]
                    right_eye = points[386]

                    dist_forehead = get_distance((hx, hy), forehead)
                    dist_left_eye = get_distance((hx, hy), left_eye)
                    dist_right_eye = get_distance((hx, hy), right_eye)

                    if (dist_left_eye < 60 or dist_right_eye < 60) and min(dist_left_eye, dist_right_eye) < dist_forehead:
                        emotion = "😭 Crying (Hand on Eyes)"
                    elif dist_forehead < 60:
                        emotion = "😓 Stressed/Tension (Hand on Forehead)"
                    elif 60 <= dist_forehead <= 120:
                        emotion = "🤔 Confused (Scratching Head)"

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show emotion label
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

    cap.release()
else:
    st.write("Please allow webcam access and press GO to start detection.") 