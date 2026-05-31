import os
import sys
import time

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from armMovement import CollaborativeRobot, State


def run_hand_move_open():
    bot = CollaborativeRobot(api=None)

    # Set the robot as already holding an object (so handoff is possible)
    held_obj = {'v1': [160, -10, -35], 'v2': [180, 10, -15], 'angle': 0.0, 'priority': 1}
    bot.held_object = held_obj
    bot.change_state(State.HOLD)

    # Hand will move through three coordinates, then open after the final coordinate
    coords = [
        {'x': 400, 'y': 0, 'z': 50, 'radius': 25.0, 'gesture': 'Closed'},  # far
        {'x': 300, 'y': 0, 'z': 5, 'radius': 25.0, 'gesture': 'Closed'},   # approaching
        {'x': 220, 'y': 0, 'z': -25, 'radius': 25.0, 'gesture': 'Closed'}, # final (near/overlap)
    ]

    print("Starting hand-move-open verification test")

    # Step through coordinates
    for i, pos in enumerate(coords):
        print(f"\n-- Move {i+1}: hand -> {pos['x'], pos['y'], pos['z']} (closed)")
        # Apply the same hand position for a few ticks to allow state transitions
        for tick in range(4):
            bot.update([pos], [held_obj])
            time.sleep(0.12)
            print(f"  Tick {tick+1}: state={bot.state}")
            if bot.state == State.EMERGENCY_STOP:
                print("Emergency stop detected. Aborting test.")
                return

    # Now open the hand at final coordinate
    open_hand = coords[-1].copy()
    open_hand['gesture'] = 'Open'
    print("\n-- Opening hand at final coordinate (gesture=Open)")

    for tick in range(6):
        bot.update([open_hand], [held_obj])
        time.sleep(0.12)
        print(f"  Open Tick {tick+1}: state={bot.state}")
        if bot.state == State.RELEASE or bot.state == State.VERIFY_RELEASE:
            print("Release sequence observed. Test successful.")
            break

    print("\nHand-move-open test complete")


if __name__ == '__main__':
    run_hand_move_open()
