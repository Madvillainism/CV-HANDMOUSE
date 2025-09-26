import time
import cv2
import mediapipe as mp

# NOTE: this file is a minimal live-stream example following the
# Google MediaPipe GestureRecognizer demo. It shows the missing pieces:
#  - a real .task model path (replace MODEL_PATH below)
#  - consistent use of the tasks API (don't reassign imported classes)
#  - creating an mp.Image from an OpenCV frame
#  - calling the recognizer in LIVE_STREAM mode with timestamp_ms

# === CONFIGURATION ===
# Path to the .task model file. Replace this with the actual .task file path.
MODEL_PATH = '/Users/joeymusante/dev/hand-remote/gesture-remote/gesture_recognizer.task'

# === SHORTCUTS TO THE TASKS API ===
# Use the 'mp.tasks' API rather than mixing imports from different helper modules.
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image


def print_result(result, output_image, timestamp_ms: int):
    """Callback for live-stream results.

    The demo prints the result; in a real app you'd use the landmarks/classification
    info to drive your logic or draw overlays on the output_image.
    """
    if result.gestures:
        print(f'gesture recognition result {result.gestures[0][0].category_name} at time {timestamp_ms}')


def main():
    # Create options for live stream mode.
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=print_result,
    )

    # Create the recognizer
    with GestureRecognizer.create_from_options(options) as recognizer:
        # Open the default camera
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print('Error: could not open camera')
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe expects RGB images. OpenCV gives BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Build an mp.Image from the numpy array. We use SRGB format.
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Timestamp in milliseconds is required for LIVE_STREAM mode.
                timestamp_ms = int(time.time() * 1000)

                # For LIVE_STREAM use the async API so results come back via the callback.
                # The exact method name can differ across versions: common names are
                # `recognize_async`, `recognize_for_video` or `detect_async`.
                # If your mediapipe version doesn't have `recognize_async`, check
                # recognizer's methods or use `recognize` for single-image mode.
                try:
                    recognizer.recognize_async(mp_image, timestamp_ms)
                except AttributeError:
                    # Fallback: if there is no recognize_async, try recognize_for_video
                    # or the synchronous recognize method. This depends on mediapipe version.
                    if hasattr(recognizer, 'recognize_for_video'):
                        recognizer.recognize_for_video(mp_image, timestamp_ms)
                    else:
                        # synchronous call (will block and return a result)
                        result = recognizer.recognize(mp_image)
                        print_result(result, None, timestamp_ms)

                # Optional: show the camera feed and stop on 'q'
                cv2.imshow('Gesture Live', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    