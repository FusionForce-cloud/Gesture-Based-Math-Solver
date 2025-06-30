import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
import cv2
import numpy as np
import time

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def count_fingers(hand_landmarks, label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    if label == "Left":
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x else 0)
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y else 0)
    return fingers.count(1)

def detectGesture(hand1_data, hand2_data):
    (hand1, label1), (hand2, label2) = hand1_data, hand2_data
    f1 = count_fingers(hand1, label1)
    f2 = count_fingers(hand2, label2)
    dist = euclidean_distance(hand1.landmark[8], hand2.landmark[8])

    if f1 == 1 and f2 == 1:
        if dist < 0.06:
            return "exit"
        return "+"
    elif (f1 == 1 and f2 == 2) or (f1 == 2 and f2 == 1):
        return "-"
    elif (f1 == 1 and f2 == 3) or (f1 == 3 and f2 == 1):
        return "*"
    elif (f1 == 1 and f2 == 4) or (f1 == 4 and f2 == 1):
        return "/"
    elif f1 == 2 and f2 == 2:
        return "del"
    elif (f1 == 1 and f2 == 5) or (f1 == 5 and f2 == 1):
        return "6"
    elif (f1 == 2 and f2 == 5) or (f1 == 5 and f2 == 2):
        return "7"
    elif (f1 == 3 and f2 == 5) or (f1 == 5 and f2 == 3):
        return "8"
    elif (f1 == 4 and f2 == 5) or (f1 == 5 and f2 == 4):
        return "9"
    elif f1 == 0 and f2 == 0:
        return "="
    elif f1 == 5 and f2 == 5:
        return "clear"
    return None

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.expression = ""
        self.result = ""
        self.last_update_time = 0
        self.delay = 1.25

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        current_time = time.time()
        hand_data = []

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = hand_handedness.classification[0].label
                hand_data.append((hand_landmarks, label))
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(hand_data) == 1:
                hand_landmarks, label = hand_data[0]
                fingers_up = count_fingers(hand_landmarks, label)
                if fingers_up in [0, 1, 2, 3, 4, 5] and current_time - self.last_update_time > self.delay:
                    self.expression += str(fingers_up)
                    self.last_update_time = current_time

            if len(hand_data) == 2:
                gesture = detectGesture(hand_data[0], hand_data[1])
                if gesture == "exit":
                    st.stop()
                elif gesture == "clear":
                    self.expression = ""
                    self.result = ""
                    self.last_update_time = current_time
                elif gesture == "del" and current_time - self.last_update_time > self.delay:
                    self.expression = self.expression[:-1]
                    self.last_update_time = current_time
                elif gesture == "=" and current_time - self.last_update_time > self.delay:
                    try:
                        self.result = str(eval(self.expression))
                    except:
                        self.result = "Error"
                    self.last_update_time = current_time
                elif gesture and current_time - self.last_update_time > self.delay:
                    self.expression += gesture
                    self.last_update_time = current_time

        cv2.putText(image, f'Expr: {self.expression}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'Result: {self.result}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Gesture-Based Math Solver")
webrtc_streamer(key="math_solver", video_processor_factory=VideoProcessor)
