import argparse
import threading
import time

from armMovement import CollaborativeRobot
from Collaborative_Robotics import dobotArm
from computer_vision import VisionSystem
from transformation import camera_to_robot, load_transform


OBJECT_PRIORITIES = {
    "blue": 2,
    "red": 1,
}

DEBUG_EVERY_SECONDS = 0.5
OBJECT_STABLE_FRAMES = 5
OBJECT_MATCH_TOLERANCE_PX = 25.0
ROBOT_WAIT_LOG_SECONDS = 2.0
ROBOT_INIT_TIMEOUT_SECONDS = 60.0


class RobotInitWorker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.api = None
        self.error = None
        self.started_at = None
        self.finished_at = None
        self._thread = None

    def start(self):
        if not self.enabled:
            self.finished_at = time.time()
            print("[INFO] Robot hardware disabled; running without Dobot.", flush=True)
            return

        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="dobot-init", daemon=True)
        self._thread.start()
        print("[INFO] Robot initialization started in background.", flush=True)

    def _run(self):
        try:
            self.api = initialize_robot()
        except BaseException as exc:
            self.error = exc
        finally:
            self.finished_at = time.time()

    def is_done(self):
        return self.finished_at is not None

    def is_ready(self):
        return self.is_done() and self.error is None

    def elapsed(self):
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class StableObjectGate:
    def __init__(self, required_frames=OBJECT_STABLE_FRAMES, tolerance_px=OBJECT_MATCH_TOLERANCE_PX):
        self.required_frames = required_frames
        self.tolerance_px = tolerance_px
        self.tracks = []
        self.next_track_id = 1

    def update(self, vision_objects):
        updated_tracks = []
        matched_track_ids = set()

        for obj in vision_objects:
            color = obj.get("color", "").lower()
            if color not in OBJECT_PRIORITIES:
                continue

            track = self._find_match(obj, matched_track_ids)
            if track is None:
                track = {
                    "id": self.next_track_id,
                    "color": color,
                    "camera": obj["camera"],
                    "count": 0,
                    "was_stable": False,
                }
                self.next_track_id += 1

            track["camera"] = obj["camera"]
            track["count"] += 1
            matched_track_ids.add(track["id"])
            updated_tracks.append(track)

        self.tracks = updated_tracks
        stable_objects = []
        for track in self.tracks:
            if track["count"] >= self.required_frames:
                if not track["was_stable"]:
                    print(
                        "[DEBUG] Object stable "
                        f"id={track['id']} color={track['color']} "
                        f"frames={track['count']} camera={track['camera']}"
                    )
                    track["was_stable"] = True
                stable_objects.append({
                    "color": track["color"],
                    "camera": track["camera"],
                })
        return stable_objects

    def debug_lines(self):
        if not self.tracks:
            return ["[DEBUG] no object candidates; passing objects=[] so IDLE is expected"]

        lines = []
        for track in self.tracks:
            status = "stable" if track["count"] >= self.required_frames else "candidate"
            lines.append(
                "[DEBUG] object "
                f"{status} id={track['id']} color={track['color']} "
                f"frames={track['count']}/{self.required_frames} camera={track['camera']}"
            )
        return lines

    def _find_match(self, obj, matched_track_ids):
        color = obj.get("color", "").lower()
        cx, cy = obj["camera"][0], obj["camera"][1]
        best_track = None
        best_dist = None

        for track in self.tracks:
            if track["id"] in matched_track_ids or track["color"] != color:
                continue
            tx, ty = track["camera"][0], track["camera"][1]
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist <= self.tolerance_px and (best_dist is None or dist < best_dist):
                best_track = track
                best_dist = dist

        return best_track


def build_robot_objects(vision_objects, transform_matrix):
    robot_objects = []
    for obj in vision_objects:
        color = obj.get("color", "").lower()
        if color not in OBJECT_PRIORITIES:
            continue

        robot_point = camera_to_robot(obj["camera"], transform_matrix)
        robot_objects.append({
            "x": robot_point[0],
            "y": robot_point[1],
            "z": robot_point[2],
            "priority": OBJECT_PRIORITIES[color],
            "color": color,
            "camera": obj["camera"],
        })
    return robot_objects


def build_robot_hands(vision_hands, transform_matrix):
    robot_hands = []
    for hand in vision_hands:
        robot_point = camera_to_robot(hand["camera"], transform_matrix)
        robot_hands.append({
            "x": robot_point[0],
            "y": robot_point[1],
            "z": robot_point[2],
            "gesture": hand.get("gesture", "Unknown"),
            "radius": hand.get("radius", 35.0),
            "camera": hand["camera"],
        })
    return robot_hands


def initialize_robot():
    start = time.time()
    print("[TIMING] robot DLL load/connect/home starting", flush=True)
    api = dobotArm.dType.load()
    print(f"[TIMING] robot DLL loaded in {time.time() - start:.2f}s", flush=True)
    dobotArm.initialize_robot(api)
    print(f"[TIMING] robot connected and homed in {time.time() - start:.2f}s", flush=True)
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)
    print(f"[TIMING] robot end effector initialized in {time.time() - start:.2f}s", flush=True)
    return api


def log_debug(detections, stable_vision_objects, robot_objects, robot_hands, bot, object_gate):
    print(
        "[DEBUG] "
        f"vision_objects={len(detections['objects'])}, "
        f"stable_vision_objects={len(stable_vision_objects)}, "
        f"vision_hands={len(detections['hands'])}, "
        f"robot_objects={len(robot_objects)}, "
        f"robot_hands={len(robot_hands)}, "
        f"state={bot.state}"
    )

    for line in object_gate.debug_lines():
        print(line)

    for obj in robot_objects:
        print(
            "[DEBUG] object "
            f"color={obj['color']} priority={obj['priority']} "
            f"camera={obj['camera']} "
            f"robot=({obj['x']:.1f}, {obj['y']:.1f}, {obj['z']:.1f})"
        )

    for hand in robot_hands:
        print(
            "[DEBUG] hand "
            f"gesture={hand['gesture']} "
            f"camera={hand['camera']} "
            f"robot=({hand['x']:.1f}, {hand['y']:.1f}, {hand['z']:.1f}) "
            f"radius={hand['radius']}"
        )


def run(use_hardware=True, show_windows=True, debug=True, require_depth=False):
    startup_start = time.time()
    print("[INFO] Starting integrated vision + robot loop.", flush=True)
    print(f"[INFO] Hardware mode: {'enabled' if use_hardware else 'disabled'}", flush=True)
    print(f"[INFO] Camera interface: {'enabled' if show_windows else 'disabled'}", flush=True)
    print("[INFO] Press Q in the camera window or Ctrl+C in the terminal to quit.", flush=True)

    transform_start = time.time()
    transform_matrix = load_transform()
    print(f"[TIMING] transform loaded in {time.time() - transform_start:.2f}s", flush=True)

    print(
        "[INFO] Starting camera "
        f"({'required depth' if require_depth else 'depth optional'}) "
        "before robot initialization.",
        flush=True,
    )
    vision_start = time.time()
    vision = VisionSystem(show_windows=show_windows, require_depth=require_depth)
    print(f"[TIMING] vision/depth startup completed in {time.time() - vision_start:.2f}s", flush=True)

    robot_worker = RobotInitWorker(enabled=use_hardware)
    robot_worker.start()
    bot = None
    object_gate = StableObjectGate()
    last_debug = 0.0
    last_wait_log = 0.0

    try:
        while True:
            detections = vision.read()
            stable_vision_objects = object_gate.update(detections["objects"])
            robot_objects = build_robot_objects(stable_vision_objects, transform_matrix)
            robot_hands = build_robot_hands(detections["hands"], transform_matrix)

            if bot is None:
                if robot_worker.error is not None:
                    raise RuntimeError(f"Robot initialization failed: {robot_worker.error}") from robot_worker.error

                if robot_worker.is_ready():
                    bot = CollaborativeRobot(api=robot_worker.api)
                    print(
                        "[INFO] Robot ready; starting robot control loop "
                        f"after {robot_worker.elapsed():.2f}s. Initial state={bot.state}",
                        flush=True,
                    )
                    print(f"[TIMING] total startup-to-robot-ready {time.time() - startup_start:.2f}s", flush=True)
                else:
                    now = time.time()
                    if robot_worker.elapsed() > ROBOT_INIT_TIMEOUT_SECONDS:
                        raise TimeoutError(
                            "Robot initialization did not finish within "
                            f"{ROBOT_INIT_TIMEOUT_SECONDS:.1f}s. Check Dobot power, COM port, and homing state."
                        )

                    if now - last_wait_log >= ROBOT_WAIT_LOG_SECONDS:
                        print(
                            "[INFO] Robot initializing... camera remains live "
                            f"({robot_worker.elapsed():.2f}s elapsed)",
                            flush=True,
                        )
                        last_wait_log = now

                    if detections["should_quit"]:
                        print("[INFO] Quit requested from camera window.")
                        break
                    time.sleep(0.03)
                    continue

            now = time.time()
            if debug and now - last_debug >= DEBUG_EVERY_SECONDS:
                log_debug(detections, stable_vision_objects, robot_objects, robot_hands, bot, object_gate)
                last_debug = now

            bot.update(robot_hands, robot_objects)

            if detections["should_quit"]:
                print("[INFO] Quit requested from camera window.")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nIntegrated run terminated by user.")
    finally:
        print("[INFO] Closing vision system.")
        vision.close()


def main():
    parser = argparse.ArgumentParser(description="Integrated camera, transform, and Dobot control loop.")
    parser.add_argument("--no-hardware", action="store_true", help="Run camera/vision loop without connecting to the Dobot.")
    parser.add_argument("--no-window", action="store_true", help="Run without OpenCV display windows.")
    parser.add_argument("--require-depth", action="store_true", help="Require OpenNI depth before starting.")
    parser.add_argument("--quiet", action="store_true", help="Disable periodic debug logs.")
    args = parser.parse_args()

    run(
        use_hardware=not args.no_hardware,
        show_windows=not args.no_window,
        debug=not args.quiet,
        require_depth=args.require_depth,
    )


if __name__ == "__main__":
    main()
