import os
import sys
import time

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Collaborative_Robotics import dobotArm
from armMovement import CollaborativeRobot, State

api = dobotArm.dType.load()


def run_three_move_scenario():
    bot = CollaborativeRobot(api=api)

    targets = [
        {'v1': [100, -10, -35], 'v2': [120, 10, -15], 'angle': 0.0, 'priority': 1},
        {'v1': [130, -10, -35], 'v2': [150, 10, -15], 'angle': 0.0, 'priority': 1},
        {'v1': [160, -10, -35], 'v2': [180, 10, -15], 'angle': 0.0, 'priority': 1},
    ]

    # Hand that will appear during the third movement (closed hand within range)
    hand_close = {'x': 165, 'y': 0, 'z': -20, 'radius': 30.0, 'gesture': 'Closed'}

    print("Starting 3-move scenario test")
    for i, target in enumerate(targets):
        print(f"\n== Movement {i+1} -> target center approx {( (target['v1'][0]+target['v2'][0])/2, (target['v1'][1]+target['v2'][1])/2 ) } ==")
        bot.target_obj = target
        bot.change_state(State.APPROACH)

        # Run a few update ticks to let the state machine progress through approach->grasp->verify->hold
        for tick in range(8):
            # On the third target, introduce the closed hand during the approach (tick >= 2)
            hands = []
            if i == 2 and tick >= 2:
                hands = [hand_close]

            bot.update(hands, [target])
            time.sleep(0.15)

            # If robot hit emergency stop, break early
            if bot.state == State.EMERGENCY_STOP:
                print("Emergency stop detected during test. Ending scenario.")
                return

    print("\n3-move scenario test complete")


if __name__ == '__main__':
    run_three_move_scenario()
