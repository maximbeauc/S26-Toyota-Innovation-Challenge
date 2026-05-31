import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

def get_gesture(hand_landmarks):
    # Heuristic-based gesture recognition
    fingers_up = 0
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    up_list = []
    for tip, pip in zip(tips, pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            up_list.append(True)
            fingers_up += 1
        else:
            up_list.append(False)
            
    if fingers_up >= 3:
        return "Open"
    elif fingers_up == 0:
        return "Closed"
    elif up_list[0] == True and up_list[1] == False and up_list[2] == False and up_list[3] == False:
        return "Pointing"
    
    return "Unknown"

latest_result = None

def process_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting hand tracking demo... Press ESC to exit.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=process_result
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        window_name = 'MediaPipe Hand Tracking Demo'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            # Check if the user closed the window
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            success, image = cap.read()
            if not success: continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Flip the image BEFORE drawing so text is not reversed
            display_image = cv2.flip(image, 1)
            h, w = display_image.shape[:2]
            
            global latest_result
            if latest_result is not None and latest_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                    for lm in hand_landmarks:
                        # Since image is flipped, mirror the x-coordinate
                        x = int((1.0 - lm.x) * w)
                        y = int(lm.y * h)
                        cv2.circle(display_image, (x, y), 3, (0, 0, 255), -1)
                        
                    gesture = get_gesture(hand_landmarks)
                    cx = int((1.0 - hand_landmarks[0].x) * w)
                    cy = int(hand_landmarks[0].y * h)
                    
                    cv2.putText(display_image, f"Hand {idx}: {gesture}", (cx, cy - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    print(f"Hand {idx} - {gesture:8} | x={(1.0 - hand_landmarks[0].x):.2f}, y={hand_landmarks[0].y:.2f}", end="\r")

            cv2.imshow(window_name, display_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
