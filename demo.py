import time
import cv2
import mediapipe as mp
import pyautogui
import math
import webbrowser
from pathlib import Path
from typing import Optional, List, Tuple
from collections import deque


# === CONFIGURATION ===
# Path to the .task model file. MUST BE placed at the repository/script root.
current_directory = Path(__file__).parent
MODEL_PATH = current_directory / 'gesture_recognizer.task'
# CURSOR CONTROL CONFIG
pyautogui.FAILSAFE = True
smooth_factor = 8

# Load and prepare the Dinho image
dinho_image = cv2.imread('dinho.png')
dinho_show = cv2.resize(dinho_image, (400, 400))

# Load and prepare the Fokiu image
fokiu_image = cv2.imread('fokiu.jpeg')
fokiu_show = cv2.resize(fokiu_image, (400, 400))

# Load and prepare the Mario image
mario_image = cv2.imread('mario.jpg')
mario_show = cv2.resize(mario_image, (400, 400))

# load hello
hello_image = cv2.imread('jelou.jfif')
hello_show = cv2.resize(hello_image, (400, 400))

# Load the scuba video
scuba_video = cv2.VideoCapture('scuba.mp4')
video_fps = 30
if video_fps == 0: # Fallback if FPS info is not available
    video_fps = 30
frame_duration = 1 / video_fps
last_video_update = 0

# Hello setup
wave_buffer = deque(maxlen=30)  # Buffer to store recent wave gesture detections
last_wave_time = 0

# CURSOR MOVEMENT SMOOTHING
min_speed = 0.01  # Minimum speed to consider for movement (to prevent jitter)
max_speed = 0.7   # Maximum speed to cap the cursor movement
alpha = 1  # Smoothing factor for cursor movement (0 = no smoothing, 1 = max smoothing)
smoothed_x = 0
smoothed_y = 0
# CLICK CONFIG
last_click = 0
click_cooldown = 0.5
freeze_cursor = False
# DRAG CONFIG
is_dragging = False
fist_start = 0
fist_confirm = 0.3  # seconds the fist must be held to confirm drag
# === SHORTCUTS TO THE TASKS API (Just makes things more readable) ===
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

# === All this nonense below is just to make the demo look cool ===
# Calibration Box
min_x, max_x = 0.3, 0.7
min_y, max_y = 0.3, 0.7
# Thumb up/down cooldown to prevent spamming the browser with multiple clicks if the gesture is held for more than a frame
_thumb_up_cooldown = 3.0  # seconds
_last_thumb_up_time = 0
# Shared state for latest gesture (simple string)
_latest_gesture: Optional[str] = None

# Latest landmarks (normalized image coords as floats 0..1)
_latest_landmarks_norm: Optional[List[Tuple[float, float]]] = None
# Standard MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (5,9),(9,10),(10,11),(11,12),   # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20),# Pinky
    (0,17)                          # Palm base connection
]
# === All this nonense above is just to make the demo look cool ===

def normalize_to_range(val, min_val, max_val):
    return max(0, min(1, (val - min_val) / (max_val - min_val)))


def is_palm_open(landmarks):
    fingers_up = [
        landmarks[8][1] < landmarks[6][1],   # Index finger
        landmarks[12][1] < landmarks[10][1], # Middle finger
        landmarks[16][1] < landmarks[14][1], # Ring finger
        landmarks[20][1] < landmarks[18][1], # Pinky
    ]

    return all(fingers_up)


def detect_wave(landmarks):

    wrist_x = landmarks[0][0]  # Wrist x
    wave_buffer.append(wrist_x)

    # We need a decent amount of frames to detect a "pattern" (e.g., 15 frames)
    if len(wave_buffer) >= 20:
        flips = 0
        for i in range(1, len(wave_buffer) - 1):
            # Check for a change in direction (Peak or Valley)
            if (wave_buffer[i-1] < wave_buffer[i] > wave_buffer[i+1]) or \
               (wave_buffer[i-1] > wave_buffer[i] < wave_buffer[i+1]):
                flips += 1
        
        # Check total distance to ensure it's not just jitter
        total_movement = max(wave_buffer) - min(wave_buffer)
        
        # A wave usually has at least 2 direction changes (Left -> Right -> Left)
        if flips >= 3 and total_movement > 0.08:
            print("wave detected")
            return True
            
    return False
def fokiu_detect(landmarks):
   # 1. Middle finger should be much higher than its base joint (extended)
    middle_extended = landmarks[12][1] < landmarks[10][1]
    
    # 2. Other fingers should be lower than their base joints (curled)
    index_curled = landmarks[8][1] > landmarks[6][1]
    ring_curled = landmarks[16][1] > landmarks[14][1]
    pinky_curled = landmarks[20][1] > landmarks[18][1]
    
    thumb_tucked = landmarks[4][1] > landmarks[8][1]

    return middle_extended and index_curled and ring_curled and pinky_curled and thumb_tucked
    #return thumb_in and pinky_in and index_curled and middle_extended and ring_curled

def dinho_detect(landmarks):
    # Landmarks: 0=Wrist, 4=Thumb, 8=Index, 12=Middle, 16=Ring, 20=Pinky
    
    # Helper to get distance from wrist
    def get_dist(p1_idx, p2_idx):
        return math.hypot(landmarks[p1_idx][0] - landmarks[p2_idx][0], 
                          landmarks[p1_idx][1] - landmarks[p2_idx][1])

    # 1. Thumb and Pinky should be FAR from wrist (Extended)
    # Using > 0.2 as a threshold (you may need to tweak this)
    thumb_extended = get_dist(4, 0) > 0.2
    pinky_extended = get_dist(20, 0) > 0.2
    
    # 2. Index, Middle, Ring should be CLOSE to wrist (Curled)
    # Using < 0.15 as a threshold
    index_curled = get_dist(8, 0) < 0.15
    middle_curled = get_dist(12, 0) < 0.15
    ring_curled = get_dist(16, 0) < 0.15
    
    # Return True only if ALL conditions are met
    return thumb_extended and pinky_extended and index_curled and middle_curled and ring_curled

def fist_detect(landmarks):
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        dist = math.hypot(landmarks[tip][0] - landmarks[0][0], landmarks[tip][1] - landmarks[0][1])
        if dist > 0.15:  # Threshold distance for open fingers
            return False
    return True

def print_result(result, output_image, timestamp_ms: int):
    """Callback for live-stream results, because its an asynchronous API we have to use a callback for its reponse. 

    The demo simpily prints the result
    You could also use the landmarks/classification info to drive your logic or draw overlays on the output_image.
    """

    text = None
    if result and result.gestures:
        # The result contains four components
        # Each component is an array
        # where each element contains the detected result of a single detected hand.
        global _last_thumb_up_time, _thumb_up_cooldown, is_dragging

        gesture = result.gestures[0][0]
        handedness = result.handedness[0][0].category_name
        text = f"{handedness} hand - {gesture.category_name} - Confidence: ({gesture.score:.2f})"

        if(gesture.category_name == 'Thumb_Down'):
            current_time = time.time()
            if current_time - _last_thumb_up_time > _thumb_up_cooldown:
                webbrowser.open('https://www.youtube.com/watch?v=tdu3vfZLe_I')
                _last_thumb_up_time = current_time

        if(gesture.category_name == 'Thumb_Up'):
            current_time = time.time()
            if current_time - _last_thumb_up_time > _thumb_up_cooldown:
                webbrowser.open('https://www.youtube.com/watch?v=F_x6_SARw30')
                _last_thumb_up_time = current_time
        else:   
            print(f'Recognized: {text}')

    # update shared state
    global _latest_gesture, _latest_landmarks_norm
    _latest_gesture = text

    # store normalized image-space landmarks (x, y)
    if result and getattr(result, 'hand_landmarks', None):
        # result.hand_landmarks is a list of lists (hands x landmarks)
        hand0 = result.hand_landmarks[0]
        _latest_landmarks_norm = [(lm.x, lm.y) for lm in hand0]
    else:
        _latest_landmarks_norm = None


def main():
    # Create options for live stream mode.
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, 
        result_callback=print_result, # Used for when recognize_async is called below asynchronously (results come back via this callback defined above) 
    )

    global freeze_cursor, last_click, is_dragging, min_x, max_x, min_y, max_y, smoothed_x, smoothed_y, alpha, fist_start, fist_confirm

    # Create the recognizer
    with GestureRecognizer.create_from_options(options) as recognizer:
        
        # Open the default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: could not open camera')
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                 # Flip the frame horizontally (so it acts like a mirror)
                frame_flipped = cv2.flip(frame, 1)   
            
                # MediaPipe expects RGB images. OpenCV gives BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Build an mp.Image from the numpy array. We use SRGB format.
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Timestamp in milliseconds is required for LIVE_STREAM mode.
                timestamp_ms = int(time.time() * 1000)

                # live image data to perform gesture recognition asynchronously
                # results for LIVE_STREAM are delivered via the `print_result` callback
                recognizer.recognize_async(mp_image, timestamp_ms)

                # Draw the latest gesture so you can see what the heck it is
                detected_gesture = _latest_gesture
                if detected_gesture:
                    cv2.putText(frame_flipped, detected_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # ===== Draw hand landmarks, this is totatly unessassry but looks cool as heck =====
                if _latest_landmarks_norm:
                    h, w = frame_flipped.shape[:2]
                    # Convert normalized landmarks to pixel coords.
                    pts = [(int((1.0 - x) * w), int(y * h)) for (x, y) in _latest_landmarks_norm]
                    # draw connections with a dark outline + colored inner line for visibility

                    norm_x, norm_y = _latest_landmarks_norm[8]# Tip of the index finger
                    thumb_x, thumb_y = _latest_landmarks_norm[4]  # Tip of the thumb

                    min_x, max_x = min(min_x, norm_x), max(max_x, norm_x)
                    min_y, max_y = min(min_y, norm_y), max(max_y, norm_y)

                    smoothed_x = (alpha * norm_x) + ((1 - alpha) * smoothed_x)
                    smoothed_y = (alpha * norm_y) + ((1 - alpha) * smoothed_y)

                    dist = math.hypot(smoothed_x - norm_x, smoothed_y - norm_y)
                    accel = 2.0 if dist > 0.05 else 1.0 # 2x speed boost if moving fast

                    mapped_x = normalize_to_range(smoothed_x, min_x, max_x) * accel
                    mapped_y = normalize_to_range(smoothed_y, min_y, max_y) * accel

                    if detected_gesture and "Victory" in detected_gesture:
                        y1, y2 = y_offset, y_offset + mario_show.shape[0]
                        x1, x2 = x_offset, x_offset + mario_show.shape[1]
                        frame_flipped[y1:y2, x1:x2] = mario_show
                        cv2.putText(frame_flipped, "Mario 64", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)   

                    dinho_sign = dinho_detect(_latest_landmarks_norm)

                    x_offset, y_offset = 50, 50

                    if dinho_sign:
                        y1, y2 = y_offset, y_offset + dinho_show.shape[0]
                        x1, x2 = x_offset, x_offset + dinho_show.shape[1]
                        frame_flipped[y1:y2, x1:x2] = dinho_show
                        cv2.putText(frame_flipped, "Ronaldinho", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


                    if fokiu_detect(_latest_landmarks_norm):
                        y1, y2 = y_offset, y_offset + fokiu_show.shape[0]
                        x1, x2 = x_offset, x_offset + fokiu_show.shape[1]
                        frame_flipped[y1:y2, x1:x2] = fokiu_show
                        cv2.putText(frame_flipped, "Fokiu", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)             

                    last_wave_time = 0
                    if detect_wave(_latest_landmarks_norm):
                        current_time = time.time()
                        if current_time - last_wave_time > 2:  # 2 second cooldown for wave gesture
                            print("Holiwi")
                            y1, y2 = y_offset, y_offset + hello_show.shape[0]
                            x1, x2 = x_offset, x_offset + hello_show.shape[1]
                            frame_flipped[y1:y2, x1:x2] = hello_show
                            cv2.putText(frame_flipped, "holi", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)  
                            """ if last_video_update > frame_duration:
                                ret, scuba_frame = scuba_video.read()
                                if not ret:
                                    scuba_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
                                    ret, scuba_frame = scuba_video.read()
                                if ret:
                                    current_video_frame = cv2.resize(scuba_frame, (200, 200))
                                    last_video_update = time.time()
                            if current_video_frame is not None:
                                h, w = current_video_frame.shape[:2]
                                frame_flipped[10:10+h, 10:10+w] = current_video_frame
                                cv2.putText(frame_flipped, "Scuba Time!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA) """
                            last_wave_time = current_time

                            

                    # Detect fist gesture for dragging
                    
                    fist_closed = fist_detect(_latest_landmarks_norm)

                    if fist_closed and not is_dragging:
                        if fist_start == 0:
                            fist_start = time.time()
                        elif time.time() - fist_start > fist_confirm:
                            pyautogui.mouseDown()
                            is_dragging = True
                            print("Drag started")

                            pyautogui.moveTo(mouse_x, mouse_y, _pause=False)

                    elif not fist_closed and is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                        print("Drag ended")
                        fist_start = 0

                    elif not fist_closed and not is_dragging:
                        fist_start = 0
                    
                    # Detect click gesture (thumb and index close together)
                    click_distance = math.hypot(norm_x - thumb_x, norm_y - thumb_y)
                    if click_distance < 0.05:  # Threshold for click gesture
                        current_time = time.time()
                        if not freeze_cursor and (current_time - last_click < 0.4):
                            freeze_cursor = True
                            pyautogui.click()
                            cv2.putText(frame_flipped, "Double Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        else:
                            pyautogui.doubleClick()
                            cv2.putText(frame_flipped, "Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        last_click = current_time
                    else:
                        freeze_cursor = False

                    screen_width, screen_height = pyautogui.size()
                    mouse_x = int((1.0 - mapped_x) * screen_width * accel)
                    mouse_y = int(mapped_y * screen_height * accel)

                    pyautogui.moveTo(mouse_x, mouse_y, _pause=False)

                    for a, b in HAND_CONNECTIONS:
                        if a < len(pts) and b < len(pts):
                            # outer outline
                            cv2.line(frame_flipped, pts[a], pts[b], (0, 0, 0), 10, cv2.LINE_AA)
                            # inner colored line
                            cv2.line(frame_flipped, pts[a], pts[b], (0, 255, 0), 6, cv2.LINE_AA)
                    # draw keypoints with outline
                    for (x_px, y_px) in pts:
                        cv2.circle(frame_flipped, (x_px, y_px), 10, (0, 0, 0), -1)
                        cv2.circle(frame_flipped, (x_px, y_px), 6, (0, 0, 255), -1)
                # ===== Draw hand landmarks, this is totatly unessassry but looks cool as heck =====

                # Show the camera feed and stop on 'q'
                cv2.imshow('Mura Cam', frame_flipped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# AUTHOR: Joey Musante - (jrmusan@gmail.com)
