import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av # Required by streamlit-webrtc for video frame handling

st.set_page_config(layout="wide")

st.title("Emotion-X: Real-time Emotion Detection")
st.write("This app detects emotions in real-time using your webcam. Please allow camera access.")
st.warning("ULTRA-OPTIMIZED DEBUG MODE: Webcam resolution reduced, hand detection disabled for live feed, and higher confidence required for face detection to conserve memory. Please check Streamlit Cloud logs for DEBUG info.")

# Initialize MediaPipe (global for drawing utilities)
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands # Still needed for Hands class, but won't be used in live processing for now
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_distance(p1, p2):
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize FaceMesh with higher confidence and no refinement to save resources
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False, # Removed refinement for performance
            min_detection_confidence=0.7, # Increased confidence
            min_tracking_confidence=0.7 # Increased confidence
        )
        # Hand model is NOT initialized here to save memory during live processing
        # self.hands = mp_hands.Hands(...) # Disabled for live processing
        
        # Face detection only for robust bounding boxes
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7) # Increased confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip for selfie view
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        emotion = "Neutral" # Default emotion
        points = [] # Initialize points for current frame

        # Process with MediaPipe Face Detection and Face Mesh
        face_detection_results = self.face_detection.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        # Hand results are not processed in the live feed for memory optimization
        # hand_results = self.hands.process(rgb_frame) # Disabled for live processing

        num_faces = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        num_hands = 0 # Hands are not processed in live feed, so set to 0 for debug print
        print(f"DEBUG: Faces detected: {num_faces}, Hands detected: {num_hands} (Hands disabled in live feed for optimization)")


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
            # Make sure these points exist before trying to access them
            if len(points) > 300: # Check if enough landmarks are detected for eyebrow points
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
            # Make sure these points exist before trying to access them
            if len(points) > 291: # Check if enough landmarks are detected for mouth points
                left_mouth = points[61]
                right_mouth = points[291]
                top_lip = points[13]
                bottom_lip = points[14]

                mouth_width = get_distance(left_mouth, right_mouth)
                mouth_height = get_distance(top_lip, bottom_lip)

                print(f"DEBUG: Mouth Width: {mouth_width}, Mouth Height: {mouth_height}")

                # Adjusted thresholds (conservative to ensure detection for debugging)
                if mouth_width > (img_w * 0.1) and mouth_height < (img_h * 0.05): # Relative to image size
                    emotion = "😊 Happy (Smiling)"
                if mouth_width > (img_w * 0.12) and mouth_height > (img_h * 0.03): # Relative to image size
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

        # Hand processing for gestures is DISABLED in live feed for memory optimization.
        # It's only enabled for uploaded images/videos.
        
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
    # ULTRA-OPTIMIZATION: Reduced resolution further, now explicitly requesting 160x120
    media_stream_constraints={"video": {"width": 160, "height": 120}, "audio": False},
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.success("Webcam is active and detecting emotions! Check the Streamlit Cloud logs for debug info.")
else:
    st.info("Webcam is initializing or paused. Please click 'Start' in the video player above and allow camera access.")


# --- Optional: File Uploader as an alternative (hand detection still enabled here) ---
st.subheader("Or Upload an Image/Video for Analysis")
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        st.write("Processing uploaded image...")
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1) # Read as BGR
        
        # --- Emotion detection logic for static images ---
        # Initialize models for static image processing - they will run hands here
        with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection_static, \
             mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True) as face_mesh_static, \
             mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7) as hands_static:
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image.shape

            face_detection_results_static = face_detection_static.process(rgb_image)
            face_results_static = face_mesh_static.process(rgb_image)
            hand_results_static = hands_static.process(rgb_image) # Corrected: using rgb_image for hands
            
            static_emotion = "Neutral (Uploaded)" # Default for static image
            points_static = [] # Initialize points for static image processing

            # Draw Face Bounding Box for uploaded image
            if face_detection_results_static.detections:
                for detection in face_detection_results_static.detections:
                    mp_drawing.draw_detection(image, detection)

            # Replicate your face detection and emotion logic here for the static image
            if face_results_static.multi_face_landmarks:
                face_landmarks_static = face_results_static.multi_face_landmarks[0]
                
                for lm in face_landmarks_static.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    points_static.append((x, y))

                # Angry detection
                if len(points_static) > 300: # Check if enough landmarks are detected
                    left_brow_left_y = points_static[70][1]
                    left_brow_right_y = points_static[52][1]
                    right_brow_left_y = points_static[282][1]
                    right_brow_right_y = points_static[300][1]

                    if right_brow_right_y < right_brow_left_y and left_brow_left_y < left_brow_right_y:
                        static_emotion = "😠 Angry (Uploaded)"
                    
                # Smile/Laugh detection
                if len(points_static) > 291: # Check if enough landmarks are detected
                    left_mouth = points_static[61]
                    right_mouth = points_static[291]
                    top_lip = points_static[13]
                    bottom_lip = points_static[14] # Corrected: using points_static

                    mouth_width = get_distance(left_mouth, right_mouth)
                    mouth_height = get_distance(top_lip, bottom_lip)

                    # Adjusted thresholds (conservative to ensure detection for debugging)
                    if mouth_width > (img_w * 0.1) and mouth_height < (img_h * 0.05):
                        static_emotion = "😊 Happy (Uploaded)"
                    if mouth_width > (img_w * 0.12) and mouth_height > (img_h * 0.03):
                        static_emotion = "😂 Laughing (Uploaded)"

                # Draw landmarks on the static image for visualization
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

            # Replicate hand detection logic for static image (still enabled for uploaded)
            if hand_results_static.multi_hand_landmarks and face_results_static.multi_face_landmarks and points_static:
                for hand_landmarks_static in hand_results_static.multi_hand_landmarks:
                    hand_tip = hand_landmarks_static.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    hx, hy = int(hand_tip.x * img_w), int(hand_tip.y * img_h)

                    if points_static: 
                        forehead = points_static[10]
                        left_eye = points_static[159]
                        right_eye = points_static[386]

                        dist_forehead = get_distance((hx, hy), forehead)
                        dist_left_eye = get_distance((hx, hy), left_eye)
                        dist_right_eye = get_distance((hx, hy), right_eye)

                        # Adjusted thresholds
                        if (dist_left_eye < (img_w * 0.08) or dist_right_eye < (img_w * 0.08)) and min(dist_left_eye, dist_right_eye) < dist_forehead:
                            static_emotion = "😭 Crying (Hand on Eyes, Uploaded)"
                        elif dist_forehead < (img_w * 0.08):
                            static_emotion = "😓 Stressed/Tension (Hand on Forehead, Uploaded)"
                        elif (img_w * 0.08) <= dist_forehead <= (img_w * 0.2):
                            static_emotion = "🤔 Confused (Scratching Head, Uploaded)"

                    mp_drawing.draw_landmarks(image, hand_landmarks_static, mp_hands.HAND_CONNECTIONS)


            cv2.putText(image, f'Emotion: {static_emotion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        st.image(image, channels="BGR", caption=f"Processed Uploaded Image: {static_emotion}")

    elif uploaded_file.type.startswith("video"):
        st.warning("Video file processing on Streamlit Cloud can be resource-intensive and may require more advanced techniques for frame-by-frame analysis. Displaying video directly.")
        st.video(uploaded_file)
