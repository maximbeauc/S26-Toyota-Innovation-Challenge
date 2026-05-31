import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

# Warning: You may need to install openni: pip install openni
# and initialize it with your local OpenNI redistributable path
try:
    from openni import openni2
except ImportError as e:
    print(f"Warning: 'openni' library import failed: {e}")
    print("Ensure you are running in the correct virtual environment.")
    openni2 = None

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

def get_depth_at_pixel(depth_frame, cx, cy, window_size=5):
    """
    Get the median depth value around a specific pixel to avoid NO_DEPTH (0) holes.
    """
    h, w = depth_frame.shape
    x_min = max(0, cx - window_size // 2)
    x_max = min(w, cx + window_size // 2 + 1)
    y_min = max(0, cy - window_size // 2)
    y_max = min(h, cy + window_size // 2 + 1)
    
    region = depth_frame[y_min:y_max, x_min:x_max]
    valid_depths = region[region > 0]
    
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    return 0.0

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Initialize OpenNI2 for Depth
    use_openni = False
    depth_stream = None
    if openni2 is not None:
        try:
            # Change this path to where your OpenNI2 binaries (Redist) are located
            # openni2.initialize('C:/Program Files/OpenNI2/Redist') # Example Windows path
            openni2.initialize() # Tries default locations
            dev = openni2.Device.open_any()
            print("Connected to Orbbec Astra Depth Sensor.")
            
            depth_stream = dev.create_depth_stream()
            depth_stream.start()
            use_openni = True
            
            # Enable depth-color sync if supported by Astra
            dev.set_depth_color_sync_enabled(True)
            
        except Exception as e:
            print(f"OpenNI2 initialization failed: {e}")
            print("Falling back to standard webcam without Z-depth...")
    
    # Astra Pro provides RGB over standard UVC
    # If index 0 is your laptop webcam, Astra RGB is usually index 1 or 2
    rgb_cap_index = 1 # Changed from 0 to 1 to target the external Orbbec camera
    
    # Optional: Automatically search for the first 3 indices to find one that isn't the laptop webcam
    # Or just stick to index 1
    cap = cv2.VideoCapture(rgb_cap_index)
    if not cap.isOpened():
        print(f"Warning: Could not open camera at index {rgb_cap_index}. Trying index 0...")
        rgb_cap_index = 0
        cap = cv2.VideoCapture(rgb_cap_index)
        
        if not cap.isOpened():
            print("Error: Could not open any webcam.")
            return

    # Camera Intrinsics (Provided in prompt for Astra Pro Plus)
    cx_cam, cy_cam = 640.0, 360.0 # From your readme (these are typical for 720p, you might need to adjust for 480p)
    fx_cam, fy_cam = 1050.0, 1050.0

    print("Starting Orbbec hand tracking... Press ESC to exit.")
    
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
        window_name = 'Astra Depth Hand Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            success, image = cap.read()
            if not success: continue

            # Get depth frame if OpenNI is active
            depth_frame = None
            if use_openni:
                frame = depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                # Astra Depth is usually 640x480 or 320x240
                depth_frame = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=frame_data)

            # NOTE: If RGB and depth resolutions differ, you must map the coordinates properly.
            # Assuming your RGB frame gives MediaPipe XY [0,1], we can scale this to the Depth frame dimensions.

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            
            # Flip image for user display
            display_image = cv2.flip(image, 1)
            h, w = display_image.shape[:2]
            
            global latest_result
            if latest_result is not None and latest_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                    # Draw points
                    for lm in hand_landmarks:
                        x_disp = int((1.0 - lm.x) * w)
                        y_disp = int(lm.y * h)
                        cv2.circle(display_image, (x_disp, y_disp), 3, (0, 0, 255), -1)
                        
                    gesture = get_gesture(hand_landmarks)
                    
                    real_x, real_y, real_z = 0.0, 0.0, 0.0
                    
                    if use_openni and depth_frame is not None:
                        dh, dw = depth_frame.shape
                        
                        # Get real coordinates (not flipped) of the wrist
                        wrist_x_pixel = int(hand_landmarks[0].x * dw)
                        wrist_y_pixel = int(hand_landmarks[0].y * dh)
                        
                        # Get real depth in mm using local median
                        z_mm = get_depth_at_pixel(depth_frame, wrist_x_pixel, wrist_y_pixel)
                        
                        if z_mm > 0:
                            # Convert mm to meters
                            real_z = z_mm / 1000.0
                            # Pinhole model math
                            real_x = (wrist_x_pixel - cx_cam) * real_z / fx_cam
                            real_y = (wrist_y_pixel - cy_cam) * real_z / fy_cam
                        
                    # Calculate flipped display coordinates
                    disp_cx = int((1.0 - hand_landmarks[0].x) * w)
                    disp_cy = int(hand_landmarks[0].y * h)
                    
                    cv2.putText(display_image, f"H{idx}: {gesture}", (disp_cx, disp_cy - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if use_openni and real_z > 0:
                        cv2.putText(display_image, f"Z: {real_z:.2f}m", (disp_cx, disp_cy - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)
                        print(f"Hand {idx} - {gesture:8} | X:{real_x:.2f}m, Y:{real_y:.2f}m, Z:{real_z:.2f}m    ", end="\r")
                    else:
                        print(f"Hand {idx} - {gesture:8} | No Depth", end="\r")

            cv2.imshow(window_name, display_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
    print("\nExiting...")
    cap.release()
    if use_openni:
        depth_stream.stop()
        openni2.unload()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
