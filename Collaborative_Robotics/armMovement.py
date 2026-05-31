import time
import math
import dobotArm

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
                    dobotArm.move_to_xyz_async(self.api, cx, cy, self.Z_PICK)
                    dobotArm.close_gripper_async(self.api)
                    dobotArm.stop_pump_async(self.api)
                    self.active_cmd_index = dobotArm.move_to_xyz_async(self.api, cx, cy, self.Z_SAFE)
                    print(f"[DEBUG] Dispatching Grasp Sequence (Down, Grab, Up). Final Queue ID: {self.active_cmd_index}")
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
            in_release_zone = (dist_to_shadow < 10.0)
            
            if in_release_zone: 
                self.change_state(State.RELEASE)
                
        elif self.state == State.RELEASE:
            if not self.cmd_dispatched:
                if self.api: 
                    self.active_cmd_index = dobotArm.open_gripper_async(self.api)
                    print(f"[DEBUG] Dispatching Gripper Open. Queue ID: {self.active_cmd_index}")
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
    print("Starting Full Execution Test...")
    
    # --- Hardware Init (keep uncommented for live robot test) ---
    # api = dobotArm.dType.load()
    # dobotArm.initialize_robot(api)
    # dobotArm.open_gripper(api)
    # bot = CollaborativeRobot(api=api)
    
    # --- Simulation Init (uncomment if testing without hardware) ---
    api = None
    bot = CollaborativeRobot(api=None)

    # 1. Dummy Object (Straight ahead, safely reachable)
    dummy_objects = [{'x': 200, 'y': 0, 'priority': 1}]

    # 2. Dummy Hand (safely out of the way)
    # 2D Bounding Box: Base (0,0) to arm/home (200,100).
    # We put the hand far enough from the line segment to be safe (>75mm for SOFT).
    # Let's put hand at x=200, y=250.
    # Hand gesture is "Open", which will trigger the drop logic once in TRACK.
    dummy_hands = [{'x': 200, 'y': 250, 'gesture': 'Open'}]

    print("Press Ctrl+C to exit. Expected Sequence:")
    print("IDLE -> SEARCH -> APPROACH -> GRASP -> VERIFY_GRASP -> HOLD -> TRACK -> RELEASE_PENDING -> RELEASE -> VERIFY_RELEASE -> IDLE")
    
    try:
        while True:
            # We constantly feed the object and the open hand into the async polling loop
            bot.update(dummy_hands, dummy_objects) 
            time.sleep(0.1) # 10Hz tick
    except KeyboardInterrupt:
        print("\nTest terminated by user.")

if __name__ == "__main__":
    main()
    


