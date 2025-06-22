import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_stream, VideoProcessorBase, WebRtcMode
import av # Required by streamlit-webrtc for video frame handling

st.set_page_config(layout="wide") # Optional: Use wide layout for better display

st.title("Emotion-X: Real-time Emotion Detection")
st.write("This app detects emotions in real-time using your webcam. Please allow camera access in your browser when prompted.")

# Initialize MediaPipe (global for drawing utilities)
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Helper function for distance calculation (remains the same)
def get_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

# --- Emotion Detection Logic encapsulated in a VideoProcessor class ---
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe models here. They will be created once per session.
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the av.VideoFrame to a NumPy array (OpenCV format BGR)
        img = frame.to_ndarray(format="bgr24")

        # Flip the image horizontally for a selfie-view display
        # This makes the user see themselves as if in a mirror
        img = cv2.flip(img, 1)

        # Convert the BGR image to RGB for MediaPipe processing
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # Process with MediaPipe Face Mesh and Hands
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        emotion = "Neutral" # Default emotion

        # Process Face Landmarks for Emotion Detection
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Extract landmark points as (x, y) tuples
            points = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                points.append((x, y))

            # --- EYEBROW DOTS (GREEN) for Angry detection ---
            # These specific points are based on MediaPipe Face Mesh indices
            # You can draw them for debugging or visualization
            cv2.circle(img, points[70], 3, (0, 255, 0), -1)    # Left eyebrow - far left
            cv2.circle(img, points[52], 3, (0, 255, 0), -1)    # Left eyebrow - middle
            cv2.circle(img, points[282], 3, (0, 255, 0), -1)   # Right eyebrow - middle
            cv2.circle(img, points[300], 3, (0, 255, 0), -1)   # Right eyebrow - far right

            # Angry detection based on eyebrow position
            # If outer eyebrow points are lower than inner, it suggests a furrowed brow (angry)
            left_brow_left_y = points[70][1]
            left_brow_right_y = points[52][1]
            right_brow_left_y = points[282][1]
            right_brow_right_y = points[300][1]

            if right_brow_right_y < right_brow_left_y and left_brow_left_y < left_brow_right_y:
                emotion = "😠 Angry (Eyebrow Shape)"

            # --- SMILE AND LAUGH DETECTION ---
            # Indices for mouth corners and top/bottom lip
            left_mouth = points[61]
            right_mouth = points[291]
            top_lip = points[13]
            bottom_lip = points[14]

            mouth_width = get_distance(left_mouth, right_mouth)
            mouth_height = get_distance(top_lip, bottom_lip)

            # Thresholds for smile and laugh (these might need fine-tuning)
            if mouth_width > 60 and mouth_height < 20: # Wide mouth, closed lips
                emotion = "😊 Happy (Smiling)"
            if mouth_width > 70 and mouth_height > 12: # Wide mouth, open lips (showing teeth)
                emotion = "😂 Laughing (Teeth showing)"

            # Draw face landmarks (optional, for visualization)
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
                connections=mp_face.FACEMESH_IRISES, # Draw iris landmarks
                landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )


        # Process Hand Landmarks for Gesture-based Emotion Detection
        # Check if both hands and face are detected for hand-to-face gestures
        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Use index finger tip landmark (landmark 8) for hand position
                hand_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hx, hy = int(hand_tip.x * img_w), int(hand_tip.y * img_h)

                # Reference points on the face
                # Ensure 'points' list from face_landmarks is populated
                if points:
                    forehead = points[10] # Top of the head/forehead
                    left_eye = points[159] # Inner corner of left eye
                    right_eye = points[386] # Inner corner of right eye

                    # Calculate distances from hand tip to face points
                    dist_forehead = get_distance((hx, hy), forehead)
                    dist_left_eye = get_distance((hx, hy), left_eye)
                    dist_right_eye = get_distance((hx, hy), right_eye)

                    # Define thresholds for gestures (these might need fine-tuning)
                    # Hand near eyes implies crying/tired
                    if (dist_left_eye < 60 or dist_right_eye < 60) and min(dist_left_eye, dist_right_eye) < dist_forehead:
                        emotion = "😭 Crying (Hand on Eyes)"
                    # Hand near forehead implies stress/tension
                    elif dist_forehead < 60:
                        emotion = "😓 Stressed/Tension (Hand on Forehead)"
                    # Hand near head, but not specifically on forehead or eyes (e.g., scratching head)
                    elif 60 <= dist_forehead <= 120:
                        emotion = "🤔 Confused (Scratching Head)"

                # Draw hand landmarks (optional, for visualization)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display emotion label on the frame
        cv2.putText(img, f'Emotion: {emotion}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Return the processed image as an av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI for the webcam component ---
st.subheader("Live Webcam Feed")
webrtc_ctx = webrtc_stream(
    key="emotion_detection_webrtc", # Unique key for this component
    mode=WebRtcMode.SENDRECV, # Send video from client, receive processed video from server
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Request video stream, no audio
    async_processing=True, # Process frames asynchronously for better performance
)

# Display status based on WebRTC context
if webrtc_ctx.state.playing:
    st.success("Webcam is active and detecting emotions!")
elif webrtc_ctx.state.stopped:
    st.info("Webcam stopped. Click 'Start' in the video player above to reactivate.")
else:
    st.info("Waiting for webcam to start... Please allow camera access in your browser.")


# --- Optional: File Uploader as an alternative (logic for processing needs to be added) ---
st.subheader("Or Upload an Image/Video for Analysis")
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        st.write("Processing uploaded image...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1) # Read as BGR
        
        # --- Add your emotion detection logic for static images here ---
        # This part is separate from the real-time stream.
        # You'll need to re-initialize MediaPipe models for static processing
        # if you want to use the same logic as the real-time stream.
        # Example (simplified):
        
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh_static, \
             mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7) as hands_static:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape

            face_results_static = face_mesh_static.process(rgb_image)
            hand_results_static = hands_static.process(rgb_image)
            
            static_emotion = "Neutral (Uploaded)" # Default for static
            
            # Replicate your face and hand detection logic here
            if face_results_static.multi_face_landmarks:
                face_landmarks_static = face_results_static.multi_face_landmarks[0]
                points_static = []
                for lm in face_landmarks_static.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    points_static.append((x, y))

                # Angry detection (example)
                left_brow_left_y = points_static[70][1]
                left_brow_right_y = points_static[52][1]
                right_brow_left_y = points_static[282][1]
                right_brow_right_y = points_static[300][1]

                if right_brow_right_y < right_brow_left_y and left_brow_left_y < left_brow_right_y:
                    static_emotion = "😠 Angry (Uploaded)"
                
                # Smile/Laugh detection (example)
                left_mouth = points_static[61]
                right_mouth = points_static[291]
                top_lip = points_static[13]
                bottom_lip = points_static[14]

                mouth_width = get_distance(left_mouth, right_mouth)
                mouth_height = get_distance(top_lip, bottom_lip)

                if mouth_width > 60 and mouth_height < 20:
                    static_emotion = "😊 Happy (Uploaded)"
                if mouth_width > 70 and mouth_height > 12:
                    static_emotion = "😂 Laughing (Uploaded)"

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_static,
                    connections=mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_static,
                    connections=mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_static,
                    connections=mp_face.FACEMESH_IRISES,
                    landmark_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

            if hand_results_static.multi_hand_landmarks and face_results_static.multi_face_landmarks and points_static:
                for hand_landmarks_static in hand_results_static.multi_hand_landmarks:
                    hand_tip = hand_landmarks_static.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    hx, hy = int(hand_tip.x * img_w), int(hand_tip.y * img_h)

                    forehead = points_static[10]
                    left_eye = points_static[159]
                    right_eye = points_static[386]

                    dist_forehead = get_distance((hx, hy), forehead)
                    dist_left_eye = get_distance((hx, hy), left_eye)
                    dist_right_eye = get_distance((hx, hy), right_eye)

                    if (dist_left_eye < 60 or dist_right_eye < 60) and min(dist_left_eye, dist_right_eye) < dist_forehead:
                        static_emotion = "😭 Crying (Hand on Eyes, Uploaded)"
                    elif dist_forehead < 60:
                        static_emotion = "😓 Stressed/Tension (Hand on Forehead, Uploaded)"
                    elif 60 <= dist_forehead <= 120:
                        static_emotion = "🤔 Confused (Scratching Head, Uploaded)"

                    mp_drawing.draw_landmarks(image, hand_landmarks_static, mp_hands.HAND_CONNECTIONS)


            cv2.putText(image, f'Emotion: {static_emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        st.image(image, channels="BGR", caption=f"Processed Uploaded Image: {static_emotion}")

    elif uploaded_file.type.startswith("video"):
        st.warning("Video file processing on Streamlit Cloud can be resource-intensive and may require more advanced techniques for frame-by-frame analysis. Displaying video directly.")
        st.video(uploaded_file) # Display the uploaded video directly
