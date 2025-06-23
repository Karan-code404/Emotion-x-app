import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av # Required by streamlit-webrtc for video frame handling

st.set_page_config(layout="wide")

st.title("Emotion-X: Real-time Emotion Detection")
st.write("This app detects emotions in real-time using your webcam. Please allow camera access.")
st.warning("OPTIMIZED DEBUG MODE: Webcam resolution increased, hand detection enabled for live feed. Please check Streamlit Cloud logs for DEBUG info to fine-tune emotion thresholds.")

# Initialize MediaPipe (global for drawing utilities)
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands # Now explicitly used for live processing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize FaceMesh with high confidence
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False, # Keeping False for now; can be enabled later if performance allows
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # Hand model is NOW initialized for live processing
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Face detection for robust bounding boxes
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip for selfie view
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        emotion = "Neutral" # Default emotion
        points = [] # Initialize points for current frame

        # Process with MediaPipe Face Detection, Face Mesh, and HANDS
        face_detection_results = self.face_detection.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame) # Hands processing re-enabled

        num_faces = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0 # Get actual hand count
        print(f"DEBUG: Faces detected: {num_faces}, Hands detected: {num_hands}")


        # Process Face Landmarks for Emotion Detection
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                points.append((x, y))

            # Drawing Face Bounding Box
            if face_detection_results.detections:
                for detection in face_detection_results.detections:
                    mp_drawing.draw_detection(img, detection)


            # EYEBROW DOTS (GREEN) for Angry detection
            # Check if enough landmarks are detected for eyebrow points
            if len(points) > 300: 
                cv2.circle(img, points[70], 5, (0, 255, 0), -1)
                cv2.circle(img, points[52], 5, (0, 255, 0), -1)
                cv2.circle(img, points[282], 5, (0, 255, 0), -1)
                cv2.circle(img, points[300], 5, (0, 255, 0), -1)

                left_brow_left_y = points[70][1]
                left_brow_right_y = points[52][1]
                right_brow_left_y = points[282][1]
                right_brow_right_y = points[300][1]

                if right_brow_right_y < right_brow_left_y and left_brow_left_y < left_brow_right_y:
                    emotion = "😠 Angry (Eyebrow Shape)"
                
                print(f"DEBUG: Eyebrow Y (L_left: {left_brow_left_y}, L_right: {left_brow_right_y}, R_left: {right_brow_left_y}, R_right: {right_brow_right_y})")


            # SMILE AND LAUGH DETECTION
            # Check if enough landmarks are detected for mouth points
            if len(points) > 291: 
                left_mouth = points[61]
                right_mouth = points[291]
                top_lip = points[13]
                bottom_lip = points[14]

                mouth_width = get_distance(left_mouth, right_mouth)
                mouth_height = get_distance(top_lip, bottom_lip)

                print(f"DEBUG: Mouth Width: {mouth_width}, Mouth Height: {mouth_height}")

                # Adjusted thresholds relative to image size for 640x480 resolution
                # These will likely need fine-tuning based on actual DEBUG values
                if mouth_width > (img_w * 0.08) and mouth_height < (img_h * 0.03):
                    emotion = "😊 Happy (Smiling)"
                if mouth_width > (img_w * 0.1) and mouth_height > (img_h * 0.02):
                    emotion = "😂 Laughing (Teeth showing)"

            # Draw face landmarks (for full visualization)
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_IRISES,
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

        # Hand processing for gestures is NOW enabled for live feed
        if hand_results.multi_hand_landmarks and num_faces > 0: # Ensure face is also detected
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hx, hy = int(hand_tip.x * img_w), int(hand_tip.y * img_h)

                if points: # Only proceed if face landmarks (points) are available
                    forehead = points[10]
                    left_eye = points[159]
                    right_eye = points[386]

                    dist_forehead = get_distance((hx, hy), forehead)
                    dist_left_eye = get_distance((hx, hy), left_eye)
                    dist_right_eye = get_distance((hx, hy), right_eye)
                    
                    print(f"DEBUG: Hand Distances (Forehead: {dist_forehead}, Left Eye: {dist_left_eye}, Right Eye: {dist_right_eye})")


                    # Adjusted thresholds relative to image size for 640x480
                    if (dist_left_eye < (img_w * 0.05) or dist_right_eye < (img_w * 0.05)) and min(dist_left_eye, dist_right_eye) < dist_forehead:
                        emotion = "😭 Crying (Hand on Eyes)"
                    elif dist_forehead < (img_w * 0.05):
                        emotion = "😓 Stressed/Tension (Hand on Forehead)"
                    elif (img_w * 0.05) <= dist_forehead <= (img_w * 0.15):
                        emotion = "🤔 Confused (Scratching Head)"

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display emotion label on the frame
        print(f"DEBUG: Final Emotion: {emotion}")
        cv2.putText(img, f'Emotion: {emotion}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI for the webcam component ---
st.subheader("Live Webcam Feed")
webrtc_ctx = webrtc_streamer(
    key="emotion_detection_webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=EmotionProcessor,
    # INCREASED RESOLUTION: Attempting 640x480 for better quality.
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("Webcam is active and detecting emotions! Check the Streamlit Cloud logs for debug info.")
else:
    st.info("Webcam is initializing or paused. Please click 'Start' in the video player above and allow camera access.")

# --- Removed the file upload option entirely ---
# No more code here for file upload.
