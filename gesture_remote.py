import cv2
import mediapipe as mp
import numpy as np

import time

from samsungtvws import SamsungTVWS

ip = "192.168.12.222"
token_file = ".samsungtv.token"


tv = SamsungTVWS(ip, port=8002, token_file=token_file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# How many consecutive frames a gesture must be detected to trigger action
GESTURE_FRAMES_REQUIRED = 120

# Track how many frames each gesture has been detected
gesture_counts = {
    "Point Right": 0,
    "Point Left": 0,
    "Enter": 0,
    "Menu": 0,
    "None": 0
}

# Some notes on MediaPipe hand landmarks:
# 0: wrist
# Each finger has 4 landmarks:

# Tip (farthest point)

# DIP (Distal Interphalangeal)

# PIP (Proximal Interphalangeal)

# MCP (Metacarpophalangeal, where finger meets palm)

    
def is_open_palm(hand_landmarks):
    """Detect if the hand is open (all fingers extended)."""
    # Check if fingertips are above their PIP joints (y is smaller higher up)
    open_fingers = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:  # index, middle, ring, pinky
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            open_fingers += 1
    return open_fingers == 4  # all four fingers extended


def is_fist(hand_landmarks):
    """Detect if the hand is closed (all fingers folded)."""
    folded_fingers = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
            folded_fingers += 1
    return folded_fingers == 4  # all four folded

    
def classify_pointing(hand_landmarks):
    """
    Classify a pointing gesture (left or right) based on MediaPipe hand landmarks.
    Returns:
        "Point Left", "Point Right", or None
    """
    # Index finger tip and pip
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    wrist = hand_landmarks.landmark[0]

    # 1. Is index finger extended? (tip higher than pip)
    index_extended = index_tip.y < index_pip.y

    # 2. Are other fingers folded?
    folded = True
    for tip, pip in [(12,10), (16,14), (20,18)]:  # middle, ring, pinky
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            folded = False

    if index_extended and folded:
        # 3. Direction: compare index tip x to wrist x
        if index_tip.x < wrist.x:
            tv.shortcuts().left()
            return "Point Left"
        else:
            tv.shortcuts().right()
            return "Point Right"
    elif is_open_palm(hand_landmarks):
        tv.shortcuts().enter()
        return "Enter"
    elif is_fist(hand_landmarks):
        tv.shortcuts().menu()
        return "Menu"
    else:
        return "None"

    return None

# Initialize video capture (webcam:1 or iphone:0)
cap = cv2.VideoCapture(1)

with mp_hands.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip and process
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected_gesture = classify_pointing(hand_landmarks)

                # ---- Gesture stability logic ----
                if detected_gesture:
                    gesture_counts[detected_gesture] += 1

                    if gesture_counts[detected_gesture] >= GESTURE_FRAMES_REQUIRED:
                        # Trigger the TV action
                        if detected_gesture == "Point Left":
                            tv.shortcuts().left()
                        elif detected_gesture == "Point Right":
                            tv.shortcuts().right()
                        elif detected_gesture == "Enter":
                            tv.shortcuts().enter()
                        elif detected_gesture == "Open menu":
                            tv.shortcuts().home()

                        # Reset count after triggering
                        gesture_counts[detected_gesture] = 0
                else:
                    # Reset all counts if no gesture detected
                    for g in gesture_counts:
                        gesture_counts[g] = 0
                                # Show gesture on screen
                cv2.putText(image, detected_gesture or "None", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Remote", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
