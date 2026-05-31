import cv2
import numpy as np
import mediapipe as mp
from enum import Enum
import json
import os
import time
import collections

try:
    from openni import openni2
    from openni import _openni2 as c_api
    _OPENNI_AVAILABLE = True
except ImportError:
    _OPENNI_AVAILABLE = False
    print("openni package not found — depth disabled. Install with: pip install openni")


# -----------------------------
# Configuration
# -----------------------------

CONFIG_PATH = "configuration.json"

DEFAULT_CONFIG = {
    "camera_index": 1,
    "min_area": 100,       # 1.5 cm block at ~1 m ≈ 8–10 px wide → ~100 px²
    "show_debug_windows": False,
    "red": {
        # Red wraps around the HSV hue wheel, so two ranges are needed.
        "range1": {"lh": 0,   "ls": 40,  "lv": 40,  "uh": 15,  "us": 255, "uv": 255},
        "range2": {"lh": 160, "ls": 40,  "lv": 40,  "uh": 180, "us": 255, "uv": 255}
    },
    # Blue: wide hue band, low saturation floor to catch slightly washed-out blues.
    "blue": {"lh": 90,  "ls": 50,  "lv": 30,  "uh": 135, "us": 255, "uv": 255},
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return DEFAULT_CONFIG.copy()


# -----------------------------
# Control Panel  (replaces OpenCV trackbars)
# -----------------------------

class ControlPanel:
    """
    Keyboard-navigable, mouse-clickable HSV slider panel.

    Keys (when the HSV Controls window has focus):
      Up / Down      — move selection
      Left / Right   — ±1
      PgUp / PgDn    — ±10
      Click bar      — jump to position
    """

    _GROUPS = [
        ("Red Range 1", (60, 60, 220), [
            ("R1_LH", "Lower H", 180), ("R1_LS", "Lower S", 255), ("R1_LV", "Lower V", 255),
            ("R1_UH", "Upper H", 180), ("R1_US", "Upper S", 255), ("R1_UV", "Upper V", 255),
        ]),
        ("Red Range 2", (90, 90, 245), [
            ("R2_LH", "Lower H", 180), ("R2_LS", "Lower S", 255), ("R2_LV", "Lower V", 255),
            ("R2_UH", "Upper H", 180), ("R2_US", "Upper S", 255), ("R2_UV", "Upper V", 255),
        ]),
        ("Blue",        (200, 130, 0), [
            ("BL_LH", "Lower H", 180), ("BL_LS", "Lower S", 255), ("BL_LV", "Lower V", 255),
            ("BL_UH", "Upper H", 180), ("BL_US", "Upper S", 255), ("BL_UV", "Upper V", 255),
        ]),
    ]

    # Windows virtual-key codes returned by cv2.waitKeyEx
    _KEY_UP    = 2490368
    _KEY_DOWN  = 2621440
    _KEY_LEFT  = 2424832
    _KEY_RIGHT = 2555904
    _KEY_PGUP  = 2162688
    _KEY_PGDN  = 2228224

    def __init__(self, initial_values: dict):
        self._flat: list[tuple[str, str, int]] = []       # (key, label, max_val)
        self._group_spans: list[tuple] = []               # (name, color, start, end)

        for name, color, sliders in self._GROUPS:
            start = len(self._flat)
            self._flat.extend(sliders)
            self._group_spans.append((name, color, start, start + len(sliders)))

        self._values = {k: initial_values.get(k, 0) for k, _, _ in self._flat}
        self._selected = 0
        self._row_map: list[tuple] = []   # filled by draw()

    # ------------------------------------------------------------------
    def get(self, key: str) -> int:
        return self._values.get(key, 0)

    # ------------------------------------------------------------------
    def handle_key(self, k: int) -> bool:
        """Returns True if the key was consumed (arrow / page keys)."""
        n = len(self._flat)
        if k == self._KEY_UP:
            self._selected = (self._selected - 1) % n
        elif k == self._KEY_DOWN:
            self._selected = (self._selected + 1) % n
        elif k == self._KEY_LEFT:
            key, _, _ = self._flat[self._selected]
            self._values[key] = max(0, self._values[key] - 1)
        elif k == self._KEY_RIGHT:
            key, _, max_val = self._flat[self._selected]
            self._values[key] = min(max_val, self._values[key] + 1)
        elif k == self._KEY_PGUP:
            key, _, _ = self._flat[self._selected]
            self._values[key] = max(0, self._values[key] - 10)
        elif k == self._KEY_PGDN:
            key, _, max_val = self._flat[self._selected]
            self._values[key] = min(max_val, self._values[key] + 10)
        else:
            return False
        return True

    # ------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            for y0, y1, key, max_val, bx, bw, idx in self._row_map:
                if y0 <= y < y1 and bx <= x <= bx + bw:
                    frac = max(0.0, min(1.0, (x - bx) / bw))
                    self._values[key] = int(round(frac * max_val))
                    self._selected = idx
                    break

    # ------------------------------------------------------------------
    def draw(self, width: int, height: int) -> np.ndarray:
        """Render the panel at the given resolution (called every frame)."""
        img = np.zeros((height, width, 3), np.uint8)
        img[:] = (28, 28, 28)

        pad       = max(6, width // 70)
        header_h  = max(22, height // 26)
        help_h    = max(16, height // 38)
        usable_h  = height - help_h - pad * 3
        n_groups  = len(self._group_spans)
        n_rows    = len(self._flat)
        row_h     = max(16, (usable_h - n_groups * header_h) // n_rows)
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fs        = max(0.28, min(0.50, row_h / 52.0))
        label_w   = width * 28 // 100
        val_col_w = width * 14 // 100

        self._row_map = []
        y = pad
        flat_idx = 0

        for group_name, color, start, end in self._group_spans:
            # Group header
            dark = tuple(max(0, c - 130) for c in color)
            cv2.rectangle(img, (0, y), (width, y + header_h), dark, -1)
            cv2.rectangle(img, (0, y), (4, y + header_h), color, -1)
            cv2.putText(img, group_name, (pad + 6, y + header_h - 5),
                        font, fs * 1.1, color, 1, cv2.LINE_AA)
            y += header_h

            for key, label, max_val in self._flat[start:end]:
                selected = flat_idx == self._selected
                bg = (50, 50, 50) if selected else (35, 35, 35)
                cv2.rectangle(img, (0, y), (width, y + row_h), bg, -1)
                if selected:
                    cv2.rectangle(img, (0, y), (4, y + row_h), (0, 210, 255), -1)

                # Label
                cv2.putText(img, label, (pad + 6, y + row_h - 5),
                            font, fs, (185, 185, 185), 1, cv2.LINE_AA)

                # Value (right-aligned)
                val = self._values[key]
                val_str = str(val)
                (tw, _), _ = cv2.getTextSize(val_str, font, fs, 1)
                cv2.putText(img, val_str, (width - tw - pad, y + row_h - 5),
                            font, fs, (240, 240, 240), 1, cv2.LINE_AA)

                # Slider bar
                bx   = pad + label_w
                bw   = width - val_col_w - bx - pad
                by   = y + row_h // 2
                bt   = max(3, row_h // 6)
                fill = int(bw * val / max_val) if max_val else 0
                bar_col = color if selected else tuple(int(c * 0.6) for c in color)

                cv2.line(img, (bx, by), (bx + bw, by), (68, 68, 68), bt)
                if fill > 0:
                    cv2.line(img, (bx, by), (bx + fill, by), bar_col, bt)
                cv2.circle(img, (bx + fill, by), max(4, bt + 1),
                           (255, 255, 255) if selected else (130, 130, 130), -1)

                self._row_map.append((y, y + row_h, key, max_val, bx, bw, flat_idx))
                y += row_h
                flat_idx += 1

        # Help strip at the bottom
        cv2.putText(
            img,
            "Up/Down: select    Left/Right: -/+1    PgUp/PgDn: -/+10    click bar to set",
            (pad, height - 4),
            font, max(0.24, fs * 0.76), (95, 95, 95), 1, cv2.LINE_AA,
        )
        return img


# Global reference so get_tb / save_config can reach the panel
_panel: ControlPanel | None = None


# -----------------------------
# Trackbar compatibility shim
# -----------------------------

def get_tb(name: str, default: int = 0) -> int:
    if _panel is not None:
        return _panel.get(name)
    return default


def save_config():
    config = {
        "camera_index": CAMERA_INDEX,
        "min_area": MIN_AREA,
        "show_debug_windows": SHOW_DEBUG_WINDOWS,
        "red": {
            "range1": {
                "lh": get_tb("R1_LH"), "ls": get_tb("R1_LS"), "lv": get_tb("R1_LV"),
                "uh": get_tb("R1_UH"), "us": get_tb("R1_US"), "uv": get_tb("R1_UV"),
            },
            "range2": {
                "lh": get_tb("R2_LH"), "ls": get_tb("R2_LS"), "lv": get_tb("R2_LV"),
                "uh": get_tb("R2_UH"), "us": get_tb("R2_US"), "uv": get_tb("R2_UV"),
            }
        },
        "blue": {
            "lh": get_tb("BL_LH"), "ls": get_tb("BL_LS"), "lv": get_tb("BL_LV"),
            "uh": get_tb("BL_UH"), "us": get_tb("BL_US"), "uv": get_tb("BL_UV"),
        },
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {CONFIG_PATH}")


# -----------------------------
# Gesture enum
# -----------------------------

class Gesture(Enum):
    CLOSED_HAND = 0
    OPEN_HAND = 1


# -----------------------------
# Gesture + palm detection
# -----------------------------

PALM_LANDMARK_IDS = [0, 5, 9, 13, 17]  # wrist + four MCP joints


def get_palm_center(hand_landmarks, frame_w, frame_h):
    xs = [hand_landmarks.landmark[i].x * frame_w for i in PALM_LANDMARK_IDS]
    ys = [hand_landmarks.landmark[i].y * frame_h for i in PALM_LANDMARK_IDS]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def fingers_up(hand_landmarks, handedness):
    """Returns [thumb, index, middle, ring, pinky] — 1 = up, 0 = down."""
    lm = hand_landmarks.landmark

    THUMB_TIP,  THUMB_IP   = 4,  3
    INDEX_TIP,  INDEX_PIP  = 8,  6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP,   RING_PIP   = 16, 14
    PINKY_TIP,  PINKY_PIP  = 20, 18

    if handedness == "Right":
        thumb_up = lm[THUMB_TIP].x < lm[THUMB_IP].x
    else:
        thumb_up = lm[THUMB_TIP].x > lm[THUMB_IP].x

    return [
        1 if thumb_up else 0,
        1 if lm[INDEX_TIP].y  < lm[INDEX_PIP].y  else 0,
        1 if lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y else 0,
        1 if lm[RING_TIP].y   < lm[RING_PIP].y   else 0,
        1 if lm[PINKY_TIP].y  < lm[PINKY_PIP].y  else 0,
    ]


def classify_gesture(fingers):
    return Gesture.OPEN_HAND if sum(fingers) >= 4 else Gesture.CLOSED_HAND


# -----------------------------
# Depth helpers
# -----------------------------

_depth_window: collections.deque = collections.deque(maxlen=5)


def read_depth_frame() -> np.ndarray | None:
    if depth_stream is None:
        return None
    frame = depth_stream.read_frame()
    if frame is None:
        return None
    buf = frame.get_buffer_as_uint16()
    return np.frombuffer(buf, dtype=np.uint16).reshape((DEPTH_HEIGHT, DEPTH_WIDTH)).copy()


def sample_palm_depth(depth_frame: np.ndarray, cx: int, cy: int, radius: int = 8) -> float | None:
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(depth_frame.shape[1], cx + radius + 1)
    y1 = min(depth_frame.shape[0], cy + radius + 1)
    valid = depth_frame[y0:y1, x0:x1]
    valid = valid[valid > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / 1000.0  # mm → metres


def smoothed_depth(raw_m: float | None) -> float | None:
    if raw_m is not None:
        _depth_window.append(raw_m)
    if not _depth_window:
        return None
    return float(np.median(list(_depth_window)))


# -----------------------------
# Colour detection helpers
# -----------------------------

def _find_all_contours(frame, display_frame, mask, min_area, label, color_bgr):
    """
    Finds every contour above min_area and draws each one.
    Returns (detected, list_of_centers, mask, result).
    """
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


def detect_red(frame, display_frame, min_area):
    """Two-range red detection to cover the hue wrap-around."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([get_tb("R1_LH"), get_tb("R1_LS"), get_tb("R1_LV")])
    upper1 = np.array([get_tb("R1_UH"), get_tb("R1_US"), get_tb("R1_UV")])
    lower2 = np.array([get_tb("R2_LH"), get_tb("R2_LS"), get_tb("R2_LV")])
    upper2 = np.array([get_tb("R2_UH"), get_tb("R2_US"), get_tb("R2_UV")])

    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return _find_all_contours(frame, display_frame, mask, min_area, "Red", (0, 0, 255))


def detect_color(frame, display_frame, prefix, label, color_bgr, min_area):
    """Single-range HSV colour detection driven by the control panel with the given prefix."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([get_tb(f"{prefix}_LH"), get_tb(f"{prefix}_LS"), get_tb(f"{prefix}_LV")])
    upper = np.array([get_tb(f"{prefix}_UH"), get_tb(f"{prefix}_US"), get_tb(f"{prefix}_UV")])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return _find_all_contours(frame, display_frame, mask, min_area, label, color_bgr)


# -----------------------------
# Load configuration
# -----------------------------

config = load_config()

CAMERA_INDEX       = config.get("camera_index", 1)
MIN_AREA           = config.get("min_area", 100)
SHOW_DEBUG_WINDOWS = config.get("show_debug_windows", False)

r1 = config["red"]["range1"]
r2 = config["red"]["range2"]
bl = config["blue"]


# -----------------------------
# Camera setup
# -----------------------------

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Could not open camera.")
    print("Try changing camera_index in configuration.json.")
    exit()


# -----------------------------
# Orbbec Astra depth stream setup
# -----------------------------

DEPTH_WIDTH  = 640
DEPTH_HEIGHT = 480
DEPTH_FPS    = 30

depth_stream = None

if _OPENNI_AVAILABLE:
    try:
        openni2.initialize(r"C:\Orbbec\OpenNI2\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs")
        _dev = openni2.Device.open_any()
        try:
            _dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            print("Depth-to-colour registration enabled.")
        except Exception as _reg_err:
            print(f"Registration not supported on this firmware, skipping: {_reg_err}")
            print("Depth values will still work — palm XY may be off by ~1–2 cm at close range.")
        depth_stream = _dev.create_depth_stream()
        depth_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX = DEPTH_WIDTH,
            resolutionY = DEPTH_HEIGHT,
            fps         = DEPTH_FPS,
        ))
        depth_stream.start()
        print("Orbbec Astra depth stream started.")
    except Exception as e:
        print(f"Could not open Orbbec depth stream: {e}")
        depth_stream = None


# -----------------------------
# MediaPipe setup
# -----------------------------

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# -----------------------------
# Control panel setup
# -----------------------------

_panel = ControlPanel({
    "R1_LH": r1["lh"], "R1_LS": r1["ls"], "R1_LV": r1["lv"],
    "R1_UH": r1["uh"], "R1_US": r1["us"], "R1_UV": r1["uv"],
    "R2_LH": r2["lh"], "R2_LS": r2["ls"], "R2_LV": r2["lv"],
    "R2_UH": r2["uh"], "R2_US": r2["us"], "R2_UV": r2["uv"],
    "BL_LH": bl["lh"], "BL_LS": bl["ls"], "BL_LV": bl["lv"],
    "BL_UH": bl["uh"], "BL_US": bl["us"], "BL_UV": bl["uv"],
})

PANEL_WIN = "HSV Controls"
cv2.namedWindow(PANEL_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(PANEL_WIN, 460, 740)
cv2.setMouseCallback(PANEL_WIN, _panel.on_mouse)


# -----------------------------
# State
# -----------------------------

palm_center  = None
palm_depth_m = None

_panel_w, _panel_h = 460, 740  # last known panel dimensions

COLOR_ROI = (440, 130, 490, 190)  # x1, y1, x2, y2 — colour detection restricted to this box

_show_debug_hsv = False
_debug_hsv: dict = {}          # {"Red": [(H,S,V),...], "Blue": ..., "Grey": ...}
_last_debug_t: float = 0.0


# -----------------------------
# Main loop  —  press S to save config, Q to quit
# -----------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # ---- Depth frame ----
    depth_frame_data = read_depth_frame()

    # ---- Hand & palm detection ----

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    hand_detected = False
    current_gesture_output = "None"
    palm_center  = None
    palm_depth_m = None

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_detected = True
        h_frame, w_frame, _ = display_frame.shape

        for hand_landmarks, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            handedness = hand_info.classification[0].label
            fingers    = fingers_up(hand_landmarks, handedness)
            gesture    = classify_gesture(fingers)

            current_gesture_output = gesture.value
            palm_center = get_palm_center(hand_landmarks, w_frame, h_frame)

            mp_draw.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            x_coords = [int(lm.x * w_frame) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h_frame) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            cv2.rectangle(display_frame,
                          (x_min - 20, y_min - 20), (x_max + 20, y_max + 20),
                          (0, 255, 0), 2)
            cv2.putText(display_frame, f"{handedness}: {gesture.value}",
                        (x_min - 20, y_min - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Fingers: {fingers}",
                        (x_min - 20, y_max + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.circle(display_frame, palm_center, 8, (0, 255, 255), -1)

            if depth_frame_data is not None:
                raw_d        = sample_palm_depth(depth_frame_data, *palm_center)
                palm_depth_m = smoothed_depth(raw_d)

            if palm_depth_m is not None:
                if palm_depth_m < 0.60:
                    depth_col = (0, 255, 80)
                elif palm_depth_m < 1.00:
                    depth_col = (0, 200, 255)
                else:
                    depth_col = (60, 60, 255)
                cv2.putText(display_frame,
                            f"Palm: {palm_center}  |  {palm_depth_m:.3f} m",
                            (palm_center[0] + 10, palm_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_col, 2)
                _bar_max_m = 2.0
                _frac      = min(1.0, palm_depth_m / _bar_max_m)
                _bar_top   = y_min - 20
                _bar_bot   = y_max + 20
                _fill_y    = int(_bar_bot - _frac * (_bar_bot - _bar_top))
                cv2.rectangle(display_frame,
                              (x_max + 25, _bar_top), (x_max + 35, _bar_bot),
                              (50, 50, 50), -1)
                cv2.rectangle(display_frame,
                              (x_max + 25, _fill_y), (x_max + 35, _bar_bot),
                              depth_col, -1)
                cv2.putText(display_frame, f"{palm_depth_m:.2f}m",
                            (x_max + 38, _bar_bot),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, depth_col, 1)
            else:
                cv2.putText(display_frame, f"Palm: {palm_center}  |  depth n/a",
                            (palm_center[0] + 10, palm_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ---- Colour detection — restricted to ROI bounding box ----

    _rx1, _ry1, _rx2, _ry2 = COLOR_ROI
    roi_frame = np.zeros_like(frame)
    roi_frame[_ry1:_ry2, _rx1:_rx2] = frame[_ry1:_ry2, _rx1:_rx2]
    cv2.rectangle(display_frame, (_rx1, _ry1), (_rx2, _ry2), (200, 200, 200), 1)

    red_detected,  red_centers,  red_mask,  red_result  = detect_red(
        roi_frame, display_frame, MIN_AREA)
    blue_detected, blue_centers, blue_mask, blue_result = detect_color(
        roi_frame, display_frame, "BL", "Blue", (255, 0, 0), MIN_AREA)

    # ---- Debug HSV sampling (once per second) ----

    _now = time.monotonic()
    if _now - _last_debug_t >= 0.1:
        _last_debug_t = _now
        _hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _debug_hsv = {}
        for _label, _centers in [
            ("Red",  red_centers),
            ("Blue", blue_centers),
        ]:
            if _centers:
                _samples = []
                for _cx, _cy in _centers:
                    _x0 = max(0, _cx - 3)
                    _y0 = max(0, _cy - 3)
                    _x1 = min(_hsv_frame.shape[1], _cx + 4)
                    _y1 = min(_hsv_frame.shape[0], _cy + 4)
                    _region = _hsv_frame[_y0:_y1, _x0:_x1]
                    if _region.size:
                        _samples.append((
                            int(np.mean(_region[:, :, 0])),
                            int(np.mean(_region[:, :, 1])),
                            int(np.mean(_region[:, :, 2])),
                        ))
                _debug_hsv[_label] = _samples

    # ---- Active detections overlay (top-left, one line per active type) ----

    active_lines = []
    if hand_detected:
        _d_str = f"{palm_depth_m:.3f} m" if palm_depth_m is not None else "depth n/a"
        active_lines.append(
            (f"Hand: {current_gesture_output} | Palm {palm_center} | {_d_str}", (0, 220, 0))
        )
    if red_detected:
        active_lines.append((f"Red  x{len(red_centers)}: {red_centers}", (60, 60, 255)))
    if blue_detected:
        active_lines.append((f"Blue x{len(blue_centers)}: {blue_centers}", (255, 80, 0)))
    if not active_lines:
        active_lines.append(("Detecting…", (160, 160, 160)))

    for i, (text, colour) in enumerate(active_lines):
        cv2.putText(display_frame, text, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2)

    cv2.putText(display_frame, "S=save  Q=quit  D=toggle HSV debug",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ---- HSV debug overlay (bottom-right) ----

    if _show_debug_hsv and _debug_hsv:
        _COLOR_BGR = {"Red": (60, 60, 255), "Blue": (255, 100, 0)}
        _lines = ["HSV debug (sampled)"]
        for _lbl, _samps in _debug_hsv.items():
            for _i, (_h, _s, _v) in enumerate(_samps):
                _suffix = f" #{_i+1}" if len(_samps) > 1 else ""
                _lines.append(f"{_lbl}{_suffix}:  H={_h:3d}  S={_s:3d}  V={_v:3d}")

        _fs, _th = 0.45, 1
        _lh = 20
        _box_w = 230
        _box_h = len(_lines) * _lh + 10
        _bx = display_frame.shape[1] - _box_w - 8
        _by = display_frame.shape[0] - _box_h - 20

        _overlay = display_frame.copy()
        cv2.rectangle(_overlay, (_bx - 4, _by - 4),
                      (_bx + _box_w, _by + _box_h), (20, 20, 20), -1)
        cv2.addWeighted(_overlay, 0.6, display_frame, 0.4, 0, display_frame)

        for _i, _line in enumerate(_lines):
            _col = (200, 200, 200) if _i == 0 else _COLOR_BGR.get(_line.split(":")[0].rstrip(" #0123456789"), (200, 200, 200))
            cv2.putText(display_frame, _line, (_bx, _by + _i * _lh + _lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs, _col, _th, cv2.LINE_AA)

    # ---- Draw control panel (resizes with the window) ----

    try:
        _r = cv2.getWindowImageRect(PANEL_WIN)
        if _r[2] > 50 and _r[3] > 50:
            _panel_w, _panel_h = _r[2], _r[3]
    except Exception:
        pass
    cv2.imshow(PANEL_WIN, _panel.draw(_panel_w, _panel_h))

    # ---- Show main window ----

    cv2.imshow("Combined Gesture + Colour Detection", display_frame)

    if SHOW_DEBUG_WINDOWS:
        for win_label, m, r in [
            ("Red",  red_mask,  red_result),
            ("Blue", blue_mask, blue_result),
        ]:
            if m is not None:
                cv2.imshow(f"{win_label} Mask", m)
            if r is not None:
                cv2.imshow(f"{win_label} Result", r)

    # ---- Key handling (waitKeyEx preserves arrow / page key codes) ----

    key = cv2.waitKeyEx(1)
    k_char = key & 0xFF

    if k_char == ord("q"):
        break
    elif k_char == ord("d"):
        _show_debug_hsv = not _show_debug_hsv
    elif k_char == ord("s"):
        save_config()
    else:
        _panel.handle_key(key)


cap.release()
hands.close()
if depth_stream is not None:
    depth_stream.stop()
if _OPENNI_AVAILABLE:
    try:
        openni2.unload()
    except Exception:
        pass
cv2.destroyAllWindows()
