# gesture_solver.py

import cv2 as cv
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# [all your other functions remain unchanged: euclidean_distance, count_fingers, detectGesture]

def run_gesture_solver():
    last_update_time = 0
    delay = 1.25
    expression = ""
    res = ""

    cap = cv.VideoCapture(0)

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv.flip(image, 1)
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
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
                if fingers_up in [0, 1, 2, 3, 4, 5] and current_time - last_update_time > delay:
                    expression += str(fingers_up)
                    last_update_time = current_time

            if len(hand_data) == 2:
                gesture = detectGesture(hand_data[0], hand_data[1])
                if gesture == "exit":
                    break
                elif gesture == "clear":
                    expression = ""
                    res = ""
                    last_update_time = current_time
                elif gesture == "del" and current_time - last_update_time > delay:
                    expression = expression[:-1]
                    last_update_time = current_time
                elif gesture == "=" and current_time - last_update_time > delay:
                    try:
                        res = str(eval(expression))
                    except:
                        res = "Error"
                    last_update_time = current_time
                elif gesture and current_time - last_update_time > delay:
                    expression += gesture
                    last_update_time = current_time

        cv.putText(image, f'Expr: {expression}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.putText(image, f'Result: {res}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        cv.imshow("Gesture Math Solver", image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            expression = ""
            res = ""

    cap.release()
    cv.destroyAllWindows()
