import time
import math
from Collaborative_Robotics import dobotArm


TEST_OBJECTS = [
    {'id': 'object_1', 'x': 220, 'y': -60, 'priority': 2},
    {'id': 'object_2', 'x': 220, 'y': 60, 'priority': 1},
]

TEST_HAND_START = {'x': 150.0, 'y': -170.0, 'z': 40.0, 'radius': 35.0, 'gesture': 'Open'}

class State:
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    VERIFY_GRASP = "VERIFY_GRASP"
    HOLD = "HOLD"
    TRACK = "TRACK"
    RELEASE_PENDING = "RELEASE_PENDING"
    RELEASE = "RELEASE"
    VERIFY_RELEASE = "VERIFY_RELEASE"
    RECOVER = "RECOVER"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class CollaborativeRobot:
    def __init__(self, api=None):
        self.api = api
        self.state = State.IDLE
        self.held_object = None
        self.target_obj = None
        
        # Configurations (Coordinate frame is Dobot space in mm)
        self.Z_SAFE = 40
        self.Z_PICK = 10 # Example pick height, adapt to your objects
        self.Z_HOVER = 50 # Height at which robot shadows the hand
        
        # Safety parameters (distances in mm)
        self.D_HARD = 35.0 # Emergency stop threshold (7cm wide bounding box -> 35mm radius)
        self.D_SOFT = 75.0 # Slowdown threshold (15cm wide bounding box -> 75mm radius)
        
        # Timing and debounce
        self.verify_grasp_time = 0.5 # Seconds to confirm pick
        self.state_enter_time = time.time()
        
        # Gesture Debouncing
        self.gesture_debounce_frames = 5
        self.open_gesture_count = 0
        
        self.cooldown_after_release = 1.0 # Seconds before picking up next object
        
        self.active_cmd_index = -1
        self.cmd_dispatched = False

    def is_robot_busy(self):
        if not self.api: return False
        # Do not use cached active_cmd_index if we don't have one
        if self.active_cmd_index == -1: return False
        return self.active_cmd_index > dobotArm.dType.GetQueuedCmdCurrentIndex(self.api)[0]
        
    def change_state(self, new_state):
        print(f"[STATE] {self.state} -> {new_state}")
        self.state = new_state
        self.state_enter_time = time.time()
        self.cmd_dispatched = False

    # ----------------------------------------------------
    # Shape-Aware Collision & Distance Math
    # ----------------------------------------------------
    def distance_point_to_line_segment_2d(self, px, py, l1x, l1y, l2x, l2y):
        """ Shortest 2D distance from a point to a line segment. """
        dx = l2x - l1x
        dy = l2y - l1y
        length_sq = dx*dx + dy*dy
        if length_sq == 0:
            # Line is just a point
            return math.hypot(px - l1x, py - l1y)
            
        # Projection of point onto the line
        t = max(0, min(1, ((px - l1x) * dx + (py - l1y) * dy) / length_sq))
        proj_x = l1x + t * dx
        proj_y = l1y + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def distance_2d(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_speed(self, dist):
        v_min, v_max = 5, 50 
        if dist > self.D_SOFT: return v_max
        elif dist > self.D_HARD: 
            return max(v_min, int(v_max * ((dist - self.D_HARD) / (self.D_SOFT - self.D_HARD))))
        return 0

    def update(self, hands, objects):
        """
        - hands: list of dicts { 'x':.., 'y':.., 'z':.., 'radius':.., 'gesture':.. }
        - objects: list of dicts { 'x':.., 'y':.., 'priority':.. }
        """
        hand = self.get_primary_hand(hands)
        
        if self.api:
            curr_pose = dobotArm.dType.GetPose(self.api)
            bot_pos = [curr_pose[0], curr_pose[1], curr_pose[2]]
        else:
            #only for simulation/testing without hardware - assume robot is at home/approach position
            bot_pos = [200, 0, 50] 
            
        dist_bot_to_hand_arm = float('inf')
        dist_bot_to_effector = float('inf')
        hand_center = None
        
        if hand:
            hand_center = [hand.get('x',200), hand.get('y',0)]
            # 2D line segment logic: Base of robot is roughly at (0, 0) in X, Y plane
            dist_bot_to_hand_arm = self.distance_point_to_line_segment_2d(
                hand_center[0], hand_center[1], 
                0, 0, bot_pos[0], bot_pos[1]
            )
            dist_bot_to_effector = self.distance_2d(hand_center, [bot_pos[0], bot_pos[1]])

        # ----------------------------------------------------
        # Scenario C: Handoff Detection
        # ----------------------------------------------------
        handoff_triggered = False
        allowed_handoff_states = [State.HOLD, State.TRACK, State.RELEASE_PENDING, State.RELEASE, State.VERIFY_RELEASE]
        if hand_center and self.held_object and (self.state in allowed_handoff_states):
            # Handoff triggered if 2D distance to end effector is within D_HARD (e.g. they reach for the claw)
            if dist_bot_to_effector <= self.D_HARD:
                handoff_triggered = True

        # ----------------------------------------------------
        # 1. Critical Mitigation: TWO-TIER SAFETY PROXIMITY STOP
        # (Exempt the hard stop if we are doing a handoff)
        # ----------------------------------------------------
        if dist_bot_to_hand_arm < self.D_HARD and self.state != State.EMERGENCY_STOP:
            if not handoff_triggered:
                print(f"!!! HARD SAFETY LIMIT TRIGGERED (dist={dist_bot_to_hand_arm:.1f}mm) !!!")
                if self.api:
                    dobotArm.dType.SetQueuedCmdStopExec(self.api)
                    dobotArm.dType.SetQueuedCmdClear(self.api)
                    self.active_cmd_index = -1
                self.change_state(State.EMERGENCY_STOP)
                return
            else:
                print(f"Safety limit bypassed: Direct user handoff detected (dist={dist_bot_to_hand_arm:.1f}mm).")

        # ----------------------------------------------------
        # 2. STATE MACHINE LOGIC
        # ----------------------------------------------------
        if self.state == State.EMERGENCY_STOP:
            if not hand or dist_bot_to_hand_arm > self.D_SOFT:
                print("Safe distance restored. Removing safety lock...")
                self.change_state(State.RECOVER)
                
        elif self.state == State.IDLE:
            if not self.cmd_dispatched:
                if self.api: 
                    self.active_cmd_index = dobotArm.move_to_home_async(self.api)
                    print(f"[DEBUG] Dispatching move_to_home_async (Queue ID: {self.active_cmd_index})")
                self.cmd_dispatched = True
                
            if objects and not self.is_robot_busy() and (time.time() - self.state_enter_time > self.cooldown_after_release):
                print("[DEBUG] Robot is idle, cooldown satisfied, hardware idle. Moving to SEARCH.")
                self.change_state(State.SEARCH)
                
        elif self.state == State.SEARCH:
            print("Searching for target object...")
            target_obj = self.get_highest_priority_object(objects)
            if target_obj:
                self.target_obj = target_obj
                self.change_state(State.APPROACH)
            else:
                self.change_state(State.IDLE)
                
        elif self.state == State.APPROACH:
            if not self.cmd_dispatched:
                print("Approaching target object...")
                if self.target_obj and self.api:
                    cx, cy = self.target_obj.get('x'), self.target_obj.get('y')
                    self.active_cmd_index = dobotArm.move_to_xyz_async(self.api, cx, cy, self.Z_SAFE)
                    print(f"[DEBUG] Dispatching Approach (X={cx:.1f}, Y={cy:.1f}, Z={self.Z_SAFE}) Queue ID: {self.active_cmd_index}")
                self.cmd_dispatched = True
                
            if not self.is_robot_busy():
                if self.target_obj:
                    print("[DEBUG] Hardware arrived at hover point. Moving to GRASP.")
                    self.change_state(State.GRASP)
                else:
                    self.change_state(State.IDLE)
                
        elif self.state == State.GRASP:
            if not self.cmd_dispatched:
                print("Attempting to grasp target object...")
                if self.target_obj and self.api:
                    cx, cy = self.target_obj.get('x'), self.target_obj.get('y')
                    down_cmd = dobotArm.move_to_xyz_async(self.api, cx, cy, self.Z_PICK)
                    close_cmd = dobotArm.close_gripper_async(self.api)
                    wait_cmd = dobotArm.wait_async(self.api, 700)
                    self.active_cmd_index = dobotArm.move_to_xyz_async(self.api, cx, cy, self.Z_SAFE)
                    print(
                        "[DEBUG] Dispatching Grasp Sequence "
                        f"(Down={down_cmd}, Close={close_cmd}, Wait={wait_cmd}, Up={self.active_cmd_index})"
                    )
                self.cmd_dispatched = True
                
            if not self.is_robot_busy():
                if self.target_obj:
                    # In VERIFY_GRASP we wait for verify_grasp_time so we have to reset timer
                    print("[DEBUG] Hardware finished grasp actions. Moving to VERIFY_GRASP.")
                    self.change_state(State.VERIFY_GRASP)
                else:
                    self.change_state(State.RECOVER)
            
        elif self.state == State.VERIFY_GRASP:
            print("Verifying grasp...")
            if time.time() - self.state_enter_time > self.verify_grasp_time:
                grasped_successfully = True
                if grasped_successfully:
                    self.held_object = self.target_obj
                    self.change_state(State.HOLD)
                else:
                    self.change_state(State.RECOVER)
                    
        elif self.state == State.HOLD:
            if handoff_triggered:
                print("Initiating handoff...")
                self.change_state(State.RELEASE_PENDING)
            elif hand: 
                print(f"[DEBUG] Hand detected at (X={hand.get('x')}, Y={hand.get('y')}). Tracking it.")
                self.change_state(State.TRACK)
                
        elif self.state == State.TRACK:
            if handoff_triggered:
                print("Initiating handoff...")
                self.change_state(State.RELEASE_PENDING)
                return

            if not hand:
                self.change_state(State.HOLD)
                return
            
            speed = self.calculate_speed(dist_bot_to_hand_arm)
            if self.api: dobotArm.dType.SetPTPCommonParams(self.api, speed, speed, isQueued=1)
            
            shadow_x = hand.get('x', 200) - 150 
            shadow_y = hand.get('y', 0)
            if self.api: 
                if not self.is_robot_busy():
                    # Only send new tracking command if previous one finished
                    self.active_cmd_index = dobotArm.move_to_xyz_async(self.api, shadow_x, shadow_y, self.Z_HOVER)
                    print(f"[DEBUG] Dispatching shadow move (X={shadow_x:.1f}, Y={shadow_y:.1f})")
            
            gesture = hand.get('gesture', 'Unknown')
            if str(gesture).lower() == 'open': self.open_gesture_count += 1
            else: self.open_gesture_count = 0
                
            if self.open_gesture_count >= self.gesture_debounce_frames:
                print(f"[DEBUG] Open hand gesture debounced ({self.open_gesture_count}/{self.gesture_debounce_frames}). Releasing.")
                self.change_state(State.RELEASE_PENDING)
                
        elif self.state == State.RELEASE_PENDING:
            if not hand:
                self.open_gesture_count = 0
                self.change_state(State.TRACK)
                return
            
            # Command shadow position
            shadow_x = hand.get('x', 200) - 150 
            shadow_y = hand.get('y', 0)
            if self.api and not self.is_robot_busy():
                self.active_cmd_index = dobotArm.move_to_xyz_async(self.api, shadow_x, shadow_y, self.Z_HOVER)
            
            # Check if we arrived
            dist_to_shadow = self.distance_2d([bot_pos[0], bot_pos[1]], [shadow_x, shadow_y])
            in_release_zone = (not self.api) or (dist_to_shadow < 10.0)
            
            if in_release_zone: 
                self.change_state(State.RELEASE)
                
        elif self.state == State.RELEASE:
            if not self.cmd_dispatched:
                if self.api: 
                    open_cmd = dobotArm.open_gripper_now(self.api)
                    dobotArm.dType.dSleep(800)
                    self.active_cmd_index = dobotArm.stop_pump_async(self.api)
                    print(
                        "[DEBUG] Dispatching Release Sequence "
                        f"(OpenNow={open_cmd}, StopPump={self.active_cmd_index})"
                    )
                self.held_object = None
                self.cmd_dispatched = True
                
            if not self.is_robot_busy():
                print("[DEBUG] Hardware opened gripper. Verification pending.")
                self.change_state(State.VERIFY_RELEASE)
            
        elif self.state == State.VERIFY_RELEASE:
            if time.time() - self.state_enter_time > 0.5: self.change_state(State.IDLE)
            
        elif self.state == State.RECOVER:
            if not self.cmd_dispatched:
                if self.api:
                    dobotArm.move_to_home_async(self.api)
                    self.active_cmd_index = dobotArm.open_gripper_async(self.api)
                self.held_object = None
                self.target_obj = None
                self.cmd_dispatched = True
                
            if not self.is_robot_busy():
                self.change_state(State.IDLE)

    def get_primary_hand(self, hands):
        if not hands: return None
        return hands[0]

    def get_highest_priority_object(self, objects):
        if not objects: return None
        return max(objects, key=lambda obj: obj.get('priority', 0), default=objects[0])


def main():
    run_two_object_robot_test()


def run_two_object_robot_test(use_hardware=True):
    """
    Live robot test:
    - Starts with two simulated objects inside the Dobot workspace.
    - Presents an open hand for the first object.
    - Keeps that hand static until the first object is dropped, then removes it.
    """
    print("Starting two-object collaborative handoff test...")
    print("Objects are simulated at reachable Dobot coordinates:")
    for obj in TEST_OBJECTS:
        print(f"  {obj['id']}: X={obj['x']}mm, Y={obj['y']}mm")
    print(
        "Open hand starts at "
        f"X={TEST_HAND_START['x']}mm, Y={TEST_HAND_START['y']}mm "
        "and disappears after the first drop."
    )

    api = None
    if use_hardware:
        api = dobotArm.dType.load()
        dobotArm.initialize_robot(api)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)

    bot = CollaborativeRobot(api=api)
    remaining_objects = [obj.copy() for obj in TEST_OBJECTS]
    hand = TEST_HAND_START.copy()
    hand_visible = True
    previous_state = bot.state

    print("Press Ctrl+C to exit.")

    try:
        while True:
            hands = [hand] if hand_visible else []
            bot.update(hands, remaining_objects)

            if previous_state == State.VERIFY_RELEASE and bot.state == State.IDLE:
                if remaining_objects:
                    completed = remaining_objects.pop(0)
                    print(f"[TEST] Completed drop for {completed['id']}.")
                    if completed['id'] == 'object_1':
                        hand_visible = False
                        print("[TEST] First object dropped. Simulated hand disappeared.")

                if not remaining_objects:
                    print("[TEST] Both objects have been picked up. Returning home and ending test.")
                    if api:
                        dobotArm.move_to_home(api)
                    break

            if previous_state == State.VERIFY_GRASP and bot.state == State.HOLD:
                if remaining_objects and remaining_objects[0]['id'] == 'object_2':
                    completed = remaining_objects.pop(0)
                    print(f"[TEST] Successfully picked up {completed['id']}.")
                    print("[TEST] Both objects have been picked up. Returning home and ending test.")
                    if api:
                        dobotArm.move_to_home(api)
                    break

            previous_state = bot.state
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest terminated by user.")

if __name__ == "__main__":
    main()
    


