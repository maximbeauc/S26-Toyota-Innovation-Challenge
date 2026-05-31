"""
main.py — integrates computer_vision, arm_movement, and the camera-to-robot
coordinate transform.

Flow each frame
---------------
1. VisionSystem detects blocks (red/blue) and the user's palm + gesture.
2. Block pixel centres + depth are transformed to robot mm via coord_transform.npy.
3. Palm pixel centre + depth is transformed the same way.
4. CollaborativeRobot.update() drives the pick-and-place state machine:
     IDLE → SEARCH → APPROACH → GRASP → HOLD → TRACK → RELEASE_PENDING → RELEASE
   The robot hovers the block directly above the palm and drops it only when the
   palm gesture has been open for several consecutive frames.
"""

import cv2
import numpy as np

from computer_vision import VisionSystem, Gesture, sample_palm_depth
from arm_movement import CollaborativeRobot

try:
    from Collaborative_Robotics import dobotArm
    _DOBOT_AVAILABLE = True
except ImportError:
    _DOBOT_AVAILABLE = False
    print("Dobot library not found — running in simulation mode (no hardware).")

TRANSFORM_PATH   = "coord_transform.npy"
FALLBACK_DEPTH_MM = 780.0   # table depth estimate when the sensor has no reading


def load_transform() -> np.ndarray | None:
    try:
        m = np.load(TRANSFORM_PATH)
        print(f"Loaded coordinate transform from {TRANSFORM_PATH}")
        return m
    except FileNotFoundError:
        print(f"WARNING: {TRANSFORM_PATH} not found — run transformation.py first. "
              "Positions will be raw pixel coords until the transform is available.")
        return None


def cam_to_robot(matrix: np.ndarray, pixel_x: int, pixel_y: int, depth_mm: float) -> np.ndarray:
    """Apply the 3×4 affine camera→robot transform. Returns [robot_x, robot_y, robot_z] in mm."""
    pt = np.array([pixel_x, pixel_y, depth_mm, 1.0], dtype=np.float64)
    return matrix @ pt


def build_objects(data: dict, transform: np.ndarray | None) -> list[dict]:
    """
    Convert detected block pixel centres to robot-space objects.
    Red blocks get higher priority (picked first).
    """
    objects = []
    for priority, (label, centers) in enumerate(
        [("blue", data["blue_centers"]), ("red", data["red_centers"])],
        start=1,
    ):
        for i, (cx, cy) in enumerate(centers):
            depth_mm = FALLBACK_DEPTH_MM
            if data["depth_frame"] is not None:
                d = sample_palm_depth(data["depth_frame"], cx, cy)
                if d is not None:
                    depth_mm = d * 1000.0

            if transform is not None:
                pos = cam_to_robot(transform, cx, cy, depth_mm)
                rx, ry = float(pos[0]), float(pos[1])
            else:
                rx, ry = float(cx), float(cy)

            objects.append({"id": f"{label}_{i}", "x": rx, "y": ry, "priority": priority})

    return objects


def build_hands(data: dict, transform: np.ndarray | None) -> list[dict]:
    """
    Convert the detected palm to a robot-space hand dict.
    Returns an empty list when no hand (or no depth) is available.
    """
    if data["palm_center"] is None or data["palm_depth_m"] is None:
        return []

    px, py = data["palm_center"]
    pz_mm  = data["palm_depth_m"] * 1000.0

    if transform is not None:
        pos = cam_to_robot(transform, px, py, pz_mm)
        rx, ry, rz = float(pos[0]), float(pos[1]), float(pos[2])
    else:
        rx, ry, rz = float(px), float(py), 0.0

    gesture_str = "Open" if data["gesture"] == Gesture.OPEN_HAND.value else "Closed"
    return [{"x": rx, "y": ry, "z": rz, "radius": 70.0, "gesture": gesture_str}]


def main():
    transform = load_transform()
    vision    = VisionSystem()

    api = None
    if _DOBOT_AVAILABLE:
        api = dobotArm.dType.load()
        dobotArm.initialize_robot(api)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)

    robot = CollaborativeRobot(api=api)
    print("System running.  Q = quit  |  S = save HSV config  |  D = HSV debug overlay")

    try:
        while True:
            data = vision.read_frame()
            if data is None:
                break

            objects = build_objects(data, transform)
            hands   = build_hands(data, transform)
            robot.update(hands, objects)

            vision.show(data, extra_overlay=f"Robot state: {robot.state}")

            key = cv2.waitKeyEx(1)
            if vision.handle_key(key) == "quit":
                break

    finally:
        vision.release()
        if api and _DOBOT_AVAILABLE:
            dobotArm.move_to_home(api)


if __name__ == "__main__":
    main()
