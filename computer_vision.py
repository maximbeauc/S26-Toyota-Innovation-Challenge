import collections
import json
import os
import time
from enum import Enum

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from openni import _openni2 as c_api
    from openni import openni2
    _OPENNI_AVAILABLE = True
except ImportError:
    try:
        from primesense import _openni2 as c_api
        from primesense import openni2
        _OPENNI_AVAILABLE = True
    except ImportError:
        c_api = None
        openni2 = None
        _OPENNI_AVAILABLE = False


CONFIG_PATH = "configuration.json"
COLOR_ROI = (440, 130, 490, 190)
OBJECT_CAMERA_Z_DEFAULT = 0.0
OPENNI_DEFAULT_PATH = r"C:\Orbbec\OpenNI2\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"

DEFAULT_CONFIG = {
    "camera_index": 1,
    "min_area": 100,
    "show_debug_windows": False,
    "red": {
        "range1": {"lh": 0, "ls": 40, "lv": 40, "uh": 15, "us": 255, "uv": 255},
        "range2": {"lh": 160, "ls": 40, "lv": 40, "uh": 180, "us": 255, "uv": 255},
    },
    "blue": {"lh": 90, "ls": 50, "lv": 30, "uh": 135, "us": 255, "uv": 255},
}


class Gesture(Enum):
    CLOSED_HAND = 0
    OPEN_HAND = 1


class ControlPanel:
    _GROUPS = [
        ("Red Range 1", (60, 60, 220), [
            ("R1_LH", "Lower H", 180), ("R1_LS", "Lower S", 255), ("R1_LV", "Lower V", 255),
            ("R1_UH", "Upper H", 180), ("R1_US", "Upper S", 255), ("R1_UV", "Upper V", 255),
        ]),
        ("Red Range 2", (90, 90, 245), [
            ("R2_LH", "Lower H", 180), ("R2_LS", "Lower S", 255), ("R2_LV", "Lower V", 255),
            ("R2_UH", "Upper H", 180), ("R2_US", "Upper S", 255), ("R2_UV", "Upper V", 255),
        ]),
        ("Blue", (200, 130, 0), [
            ("BL_LH", "Lower H", 180), ("BL_LS", "Lower S", 255), ("BL_LV", "Lower V", 255),
            ("BL_UH", "Upper H", 180), ("BL_US", "Upper S", 255), ("BL_UV", "Upper V", 255),
        ]),
    ]

    _KEY_UP = 2490368
    _KEY_DOWN = 2621440
    _KEY_LEFT = 2424832
    _KEY_RIGHT = 2555904
    _KEY_PGUP = 2162688
    _KEY_PGDN = 2228224

    def __init__(self, initial_values):
        self._flat = []
        self._group_spans = []
        for name, color, sliders in self._GROUPS:
            start = len(self._flat)
            self._flat.extend(sliders)
            self._group_spans.append((name, color, start, start + len(sliders)))
        self._values = {key: initial_values.get(key, 0) for key, _, _ in self._flat}
        self._selected = 0
        self._row_map = []

    def get(self, key, default=0):
        return self._values.get(key, default)

    def handle_key(self, key_code):
        n = len(self._flat)
        if key_code == self._KEY_UP:
            self._selected = (self._selected - 1) % n
        elif key_code == self._KEY_DOWN:
            self._selected = (self._selected + 1) % n
        elif key_code == self._KEY_LEFT:
            key, _, _ = self._flat[self._selected]
            self._values[key] = max(0, self._values[key] - 1)
        elif key_code == self._KEY_RIGHT:
            key, _, max_val = self._flat[self._selected]
            self._values[key] = min(max_val, self._values[key] + 1)
        elif key_code == self._KEY_PGUP:
            key, _, _ = self._flat[self._selected]
            self._values[key] = max(0, self._values[key] - 10)
        elif key_code == self._KEY_PGDN:
            key, _, max_val = self._flat[self._selected]
            self._values[key] = min(max_val, self._values[key] + 10)
        else:
            return False
        return True

    def on_mouse(self, event, x, y, flags, _param):
        dragging = event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)
        if event != cv2.EVENT_LBUTTONDOWN and not dragging:
            return
        for y0, y1, key, max_val, bx, bw, idx in self._row_map:
            if y0 <= y < y1 and bx <= x <= bx + bw:
                frac = max(0.0, min(1.0, (x - bx) / bw))
                self._values[key] = int(round(frac * max_val))
                self._selected = idx
                break

    def draw(self, width, height):
        img = np.zeros((height, width, 3), np.uint8)
        img[:] = (28, 28, 28)
        pad = max(6, width // 70)
        header_h = max(22, height // 26)
        help_h = max(16, height // 38)
        usable_h = height - help_h - pad * 3
        row_h = max(16, (usable_h - len(self._group_spans) * header_h) // len(self._flat))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.28, min(0.50, row_h / 52.0))
        label_w = width * 28 // 100
        val_col_w = width * 14 // 100
        self._row_map = []
        y = pad
        flat_idx = 0

        for group_name, color, start, end in self._group_spans:
            dark = tuple(max(0, c - 130) for c in color)
            cv2.rectangle(img, (0, y), (width, y + header_h), dark, -1)
            cv2.rectangle(img, (0, y), (4, y + header_h), color, -1)
            cv2.putText(img, group_name, (pad + 6, y + header_h - 5), font, fs * 1.1, color, 1, cv2.LINE_AA)
            y += header_h

            for key, label, max_val in self._flat[start:end]:
                selected = flat_idx == self._selected
                cv2.rectangle(img, (0, y), (width, y + row_h), (50, 50, 50) if selected else (35, 35, 35), -1)
                if selected:
                    cv2.rectangle(img, (0, y), (4, y + row_h), (0, 210, 255), -1)
                cv2.putText(img, label, (pad + 6, y + row_h - 5), font, fs, (185, 185, 185), 1, cv2.LINE_AA)

                val = self._values[key]
                val_str = str(val)
                (tw, _), _ = cv2.getTextSize(val_str, font, fs, 1)
                cv2.putText(img, val_str, (width - tw - pad, y + row_h - 5), font, fs, (240, 240, 240), 1, cv2.LINE_AA)

                bx = pad + label_w
                bw = width - val_col_w - bx - pad
                by = y + row_h // 2
                bt = max(3, row_h // 6)
                fill = int(bw * val / max_val) if max_val else 0
                bar_col = color if selected else tuple(int(c * 0.6) for c in color)
                cv2.line(img, (bx, by), (bx + bw, by), (68, 68, 68), bt)
                if fill > 0:
                    cv2.line(img, (bx, by), (bx + fill, by), bar_col, bt)
                cv2.circle(img, (bx + fill, by), max(4, bt + 1), (255, 255, 255) if selected else (130, 130, 130), -1)
                self._row_map.append((y, y + row_h, key, max_val, bx, bw, flat_idx))
                y += row_h
                flat_idx += 1

        cv2.putText(img, "Up/Down: select    Left/Right: -/+1    PgUp/PgDn: -/+10    click bar to set",
                    (pad, height - 4), font, max(0.24, fs * 0.76), (95, 95, 95), 1, cv2.LINE_AA)
        return img


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return DEFAULT_CONFIG.copy()


def fingers_up(hand_landmarks, handedness):
    lm = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    thumb_up = lm[4].x < lm[3].x if handedness == "Right" else lm[4].x > lm[3].x
    return [
        1 if thumb_up else 0,
        1 if lm[8].y < lm[6].y else 0,
        1 if lm[12].y < lm[10].y else 0,
        1 if lm[16].y < lm[14].y else 0,
        1 if lm[20].y < lm[18].y else 0,
    ]


def classify_gesture(fingers):
    return Gesture.OPEN_HAND if sum(fingers) >= 4 else Gesture.CLOSED_HAND


def get_palm_center(hand_landmarks, frame_w, frame_h):
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    ids = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x * frame_w for i in ids]
    ys = [landmarks[i].y * frame_h for i in ids]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


class SolutionsHandTracker:
    def __init__(self):
        hands_module, draw_module, styles_module = self._load_solutions()
        self.hands_module = hands_module
        self.draw = draw_module
        self.styles = styles_module
        self.hands = self.hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        print("[INFO] MediaPipe hand tracking: solutions API.")

    def _load_solutions(self):
        if mp is None:
            raise RuntimeError("MediaPipe is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")

        try:
            return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles
        except AttributeError:
            from mediapipe.python.solutions import drawing_styles, drawing_utils, hands
            return hands, drawing_utils, drawing_styles

    def detect(self, frame, display_frame, depth_sampler):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        detected_hands = []

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detected_hands

        h_frame, w_frame, _ = display_frame.shape
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = hand_info.classification[0].label
            gesture = classify_gesture(fingers_up(hand_landmarks, handedness))
            palm_center = get_palm_center(hand_landmarks, w_frame, h_frame)
            depth_mm = depth_sampler(*palm_center)
            z = depth_mm if depth_mm is not None else 0.0

            self.draw.draw_landmarks(
                display_frame,
                hand_landmarks,
                self.hands_module.HAND_CONNECTIONS,
                self.styles.get_default_hand_landmarks_style(),
                self.styles.get_default_hand_connections_style(),
            )
            cv2.circle(display_frame, palm_center, 8, (0, 255, 255), -1)
            cv2.putText(display_frame, f"{handedness}: {gesture.name.replace('_HAND', '').title()}",
                        (palm_center[0] + 10, palm_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_hands.append({
                "camera": [float(palm_center[0]), float(palm_center[1]), float(z)],
                "gesture": "Open" if gesture == Gesture.OPEN_HAND else "Closed",
                "radius": 35.0,
            })
        return detected_hands

    def close(self):
        self.hands.close()


class TasksHandTracker:
    def __init__(self):
        if mp is None:
            raise RuntimeError("MediaPipe is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")

        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "This MediaPipe install exposes neither solutions nor tasks hand tracking. "
                "Try: python -m pip install --upgrade --force-reinstall mediapipe"
            ) from exc

        model_path = os.path.join(os.path.dirname(__file__), "Collaborative_Robotics", "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise RuntimeError(f"MediaPipe hand model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.vision = vision
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0
        print("[INFO] MediaPipe hand tracking: tasks API.")

    def detect(self, frame, display_frame, depth_sampler):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        detected_hands = []

        if not result.hand_landmarks:
            return detected_hands

        h_frame, w_frame, _ = display_frame.shape
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            for lm in hand_landmarks:
                x = int(lm.x * w_frame)
                y = int(lm.y * h_frame)
                cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)

            gesture = classify_gesture(fingers_up(hand_landmarks, "Right"))
            palm_center = get_palm_center(hand_landmarks, w_frame, h_frame)
            depth_mm = depth_sampler(*palm_center)
            z = depth_mm if depth_mm is not None else 0.0
            gesture_text = "Open" if gesture == Gesture.OPEN_HAND else "Closed"

            cv2.circle(display_frame, palm_center, 8, (0, 255, 255), -1)
            cv2.putText(display_frame, f"Hand {idx}: {gesture_text}",
                        (palm_center[0] + 10, palm_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_hands.append({
                "camera": [float(palm_center[0]), float(palm_center[1]), float(z)],
                "gesture": gesture_text,
                "radius": 35.0,
            })
        return detected_hands

    def close(self):
        self.landmarker.close()


def create_hand_tracker():
    if mp is None:
        raise RuntimeError("MediaPipe is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")

    try:
        return SolutionsHandTracker()
    except (AttributeError, ImportError):
        return TasksHandTracker()


def load_mediapipe_solutions():
    try:
        return create_hand_tracker()
    except RuntimeError:
        raise


def _unused_load_mediapipe_solutions_compat():
    if mp is None:
        raise RuntimeError("MediaPipe is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")

    try:
        return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles
    except AttributeError:
        try:
            from mediapipe.python.solutions import drawing_styles, drawing_utils, hands
        except ImportError as exc:
            raise RuntimeError(
                "This MediaPipe install does not expose the hand-tracking solutions API. "
                "Install the supported package with: python -m pip install --upgrade --force-reinstall mediapipe"
            ) from exc
        return hands, drawing_utils, drawing_styles


def find_all_contours(frame, display_frame, mask, min_area, label, color_bgr):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        centers.append((cx, cy))
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.circle(display_frame, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(display_frame, f"{label}({cx},{cy})", (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return len(centers) > 0, centers, mask, result


class VisionSystem:
    def __init__(self, show_windows=True, require_depth=True):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")
        if np is None:
            raise RuntimeError("NumPy is required for VisionSystem. Install dependencies with: python -m pip install -r requirements.txt")

        self.config = load_config()
        self.camera_index = self.config.get("camera_index", 1)
        self.min_area = self.config.get("min_area", 100)
        self.show_debug_windows = self.config.get("show_debug_windows", False)
        self.show_windows = show_windows
        self.require_depth = require_depth
        self.depth_window = collections.deque(maxlen=5)
        self.depth_stream = None
        self.panel_w = 460
        self.panel_h = 740
        self.show_debug_hsv = False
        self.debug_hsv = {}
        self.last_debug_t = 0.0

        red1 = self.config["red"]["range1"]
        red2 = self.config["red"]["range2"]
        blue = self.config["blue"]
        self.panel = ControlPanel({
            "R1_LH": red1["lh"], "R1_LS": red1["ls"], "R1_LV": red1["lv"],
            "R1_UH": red1["uh"], "R1_US": red1["us"], "R1_UV": red1["uv"],
            "R2_LH": red2["lh"], "R2_LS": red2["ls"], "R2_LV": red2["lv"],
            "R2_UH": red2["uh"], "R2_US": red2["us"], "R2_UV": red2["uv"],
            "BL_LH": blue["lh"], "BL_LS": blue["ls"], "BL_LV": blue["lv"],
            "BL_UH": blue["uh"], "BL_US": blue["us"], "BL_UV": blue["uv"],
        })

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Try changing camera_index in configuration.json.")

        self._setup_depth()
        self.hand_tracker = create_hand_tracker()

        if self.show_windows:
            print("[INFO] Opening camera interface windows.")
            cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HSV Controls", self.panel_w, self.panel_h)
            cv2.setMouseCallback("HSV Controls", self.panel.on_mouse)

    def _setup_depth(self):
        if not _OPENNI_AVAILABLE:
            message = (
                "OpenNI Python bindings not found. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            )
            if self.require_depth:
                print(f"[ERROR] {message}")
                raise RuntimeError(message)
            print(f"{message} - depth disabled.")
            return
        try:
            self._initialize_openni()
            dev = openni2.Device.open_any()
            try:
                dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
                print("Depth-to-colour registration enabled.")
            except Exception as reg_err:
                print(f"Registration not supported on this firmware, skipping: {reg_err}")
            self.depth_stream = dev.create_depth_stream()
            self.depth_stream.set_video_mode(c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=640,
                resolutionY=480,
                fps=30,
            ))
            self.depth_stream.start()
            print("Orbbec Astra depth stream started.")
        except Exception as err:
            self.depth_stream = None
            if self.require_depth:
                print(f"[ERROR] Could not open Orbbec depth stream: {err}")
                raise RuntimeError(f"Could not open Orbbec depth stream: {err}") from err
            print(f"Could not open Orbbec depth stream: {err}")

    def _initialize_openni(self):
        candidates = []
        env_path = os.environ.get("OPENNI2_REDIST") or os.environ.get("OPENNI2_LIB")
        if env_path:
            candidates.append(env_path)
        candidates.append(OPENNI_DEFAULT_PATH)

        errors = []
        for candidate in candidates:
            if not candidate or not os.path.exists(candidate):
                continue
            try:
                openni2.initialize(candidate)
                print(f"[INFO] OpenNI2 initialized from {candidate}")
                return
            except Exception as err:
                errors.append(f"{candidate}: {err}")

        try:
            openni2.initialize()
            print("[INFO] OpenNI2 initialized from default search paths.")
            return
        except Exception as err:
            errors.append(f"default search paths: {err}")

        raise RuntimeError("OpenNI2 initialization failed. " + " | ".join(errors))

    def get_tb(self, name, default=0):
        return self.panel.get(name, default)

    def save_config(self):
        config = {
            "camera_index": self.camera_index,
            "min_area": self.min_area,
            "show_debug_windows": self.show_debug_windows,
            "red": {
                "range1": {"lh": self.get_tb("R1_LH"), "ls": self.get_tb("R1_LS"), "lv": self.get_tb("R1_LV"),
                           "uh": self.get_tb("R1_UH"), "us": self.get_tb("R1_US"), "uv": self.get_tb("R1_UV")},
                "range2": {"lh": self.get_tb("R2_LH"), "ls": self.get_tb("R2_LS"), "lv": self.get_tb("R2_LV"),
                           "uh": self.get_tb("R2_UH"), "us": self.get_tb("R2_US"), "uv": self.get_tb("R2_UV")},
            },
            "blue": {"lh": self.get_tb("BL_LH"), "ls": self.get_tb("BL_LS"), "lv": self.get_tb("BL_LV"),
                     "uh": self.get_tb("BL_UH"), "us": self.get_tb("BL_US"), "uv": self.get_tb("BL_UV")},
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {CONFIG_PATH}")

    def read_depth_frame(self):
        if self.depth_stream is None:
            return None
        frame = self.depth_stream.read_frame()
        if frame is None:
            return None
        buf = frame.get_buffer_as_uint16()
        return np.frombuffer(buf, dtype=np.uint16).reshape((480, 640)).copy()

    def sample_depth(self, depth_frame, cx, cy, radius=8):
        if depth_frame is None:
            return None
        x0 = max(0, cx - radius)
        y0 = max(0, cy - radius)
        x1 = min(depth_frame.shape[1], cx + radius + 1)
        y1 = min(depth_frame.shape[0], cy + radius + 1)
        valid = depth_frame[y0:y1, x0:x1]
        valid = valid[valid > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def smoothed_depth(self, raw_mm):
        if raw_mm is not None:
            self.depth_window.append(raw_mm)
        if not self.depth_window:
            return None
        return float(np.median(list(self.depth_window)))

    def detect_red(self, frame, display_frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.get_tb("R1_LH"), self.get_tb("R1_LS"), self.get_tb("R1_LV")])
        upper1 = np.array([self.get_tb("R1_UH"), self.get_tb("R1_US"), self.get_tb("R1_UV")])
        lower2 = np.array([self.get_tb("R2_LH"), self.get_tb("R2_LS"), self.get_tb("R2_LV")])
        upper2 = np.array([self.get_tb("R2_UH"), self.get_tb("R2_US"), self.get_tb("R2_UV")])
        mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return find_all_contours(frame, display_frame, mask, self.min_area, "Red", (0, 0, 255))

    def detect_blue(self, frame, display_frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.get_tb("BL_LH"), self.get_tb("BL_LS"), self.get_tb("BL_LV")])
        upper = np.array([self.get_tb("BL_UH"), self.get_tb("BL_US"), self.get_tb("BL_UV")])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return find_all_contours(frame, display_frame, mask, self.min_area, "Blue", (255, 0, 0))

    def _detect_hands(self, frame, display_frame, depth_frame):
        def depth_sampler(cx, cy):
            return self.smoothed_depth(self.sample_depth(depth_frame, cx, cy))

        return self.hand_tracker.detect(frame, display_frame, depth_sampler)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read camera frame.")

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        depth_frame = self.read_depth_frame()
        detected_hands = self._detect_hands(frame, display_frame, depth_frame)

        rx1, ry1, rx2, ry2 = COLOR_ROI
        roi_frame = np.zeros_like(frame)
        roi_frame[ry1:ry2, rx1:rx2] = frame[ry1:ry2, rx1:rx2]
        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (200, 200, 200), 1)

        red_detected, red_centers, red_mask, red_result = self.detect_red(roi_frame, display_frame)
        blue_detected, blue_centers, blue_mask, blue_result = self.detect_blue(roi_frame, display_frame)

        objects = []
        for color, centers in (("blue", blue_centers), ("red", red_centers)):
            for cx, cy in centers:
                z = self.sample_depth(depth_frame, cx, cy)
                objects.append({
                    "color": color,
                    "camera": [float(cx), float(cy), float(z if z is not None else OBJECT_CAMERA_Z_DEFAULT)],
                })

        active_lines = []
        if detected_hands:
            active_lines.append((f"Hands x{len(detected_hands)}", (0, 220, 0)))
        if red_detected:
            active_lines.append((f"Red x{len(red_centers)}: {red_centers}", (60, 60, 255)))
        if blue_detected:
            active_lines.append((f"Blue x{len(blue_centers)}: {blue_centers}", (255, 80, 0)))
        if not active_lines:
            active_lines.append(("Detecting...", (160, 160, 160)))
        for i, (text, color) in enumerate(active_lines):
            cv2.putText(display_frame, text, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        cv2.putText(display_frame, "S=save  Q=quit  D=toggle HSV debug",
                    (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        key = -1
        if self.show_windows:
            try:
                rect = cv2.getWindowImageRect("HSV Controls")
                if rect[2] > 50 and rect[3] > 50:
                    self.panel_w, self.panel_h = rect[2], rect[3]
            except Exception:
                pass
            cv2.imshow("HSV Controls", self.panel.draw(self.panel_w, self.panel_h))
            cv2.imshow("Combined Gesture + Colour Detection", display_frame)
            if self.show_debug_windows:
                for win_label, mask, result in (("Red", red_mask, red_result), ("Blue", blue_mask, blue_result)):
                    cv2.imshow(f"{win_label} Mask", mask)
                    cv2.imshow(f"{win_label} Result", result)

            key = cv2.waitKeyEx(1)
            key_char = key & 0xFF
            if key_char == ord("s"):
                self.save_config()
            elif key_char == ord("d"):
                self.show_debug_hsv = not self.show_debug_hsv
            else:
                self.panel.handle_key(key)

        return {
            "objects": objects,
            "hands": detected_hands,
            "key": key,
            "should_quit": (key & 0xFF) == ord("q") if key != -1 else False,
            "frame": display_frame,
        }

    def close(self):
        self.cap.release()
        self.hand_tracker.close()
        if self.depth_stream is not None:
            self.depth_stream.stop()
        if _OPENNI_AVAILABLE:
            try:
                openni2.unload()
            except Exception:
                pass
        if self.show_windows:
            cv2.destroyAllWindows()


def main():
    vision = VisionSystem(show_windows=True)
    try:
        while True:
            detections = vision.read()
            if detections["should_quit"]:
                break
    finally:
        vision.close()


if __name__ == "__main__":
    main()
