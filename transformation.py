DEFAULT_TRANSFORM_PATH = "coord_transform.npy"

CAMERA_POINTS = [
    [386, 241, 820],
    [281, 261, 830],
    [169, 220, 830],
    [223, 187, 724],
    [454, 168, 690],
]

ROBOT_POINTS = [
    [259, -97, -6],
    [221, 46, 0.99],
    [249, 214, 4.35],
    [216, 135, 110],
    [211, -160, 134],
]


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for coordinate transformation. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return np


def estimate_transform(camera_points=CAMERA_POINTS, robot_points=ROBOT_POINTS):
    import cv2
    np = require_numpy()

    camera_points = np.array(camera_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)
    return_value, matrix, inliers = cv2.estimateAffine3D(camera_points, robot_points)
    if not return_value:
        raise RuntimeError("Could not estimate camera-to-robot transform.")
    return matrix, inliers


def save_transform(path=DEFAULT_TRANSFORM_PATH):
    matrix, _ = estimate_transform()
    np.save(path, matrix)
    return matrix


def load_transform(path=DEFAULT_TRANSFORM_PATH):
    np = require_numpy()
    matrix = np.load(path)
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected transform shape (3, 4), got {matrix.shape}.")
    return matrix


def camera_to_robot(point, matrix):
    np = require_numpy()
    if len(point) != 3:
        raise ValueError("Camera point must contain exactly 3 values: x, y, z.")

    camera_point = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    robot_point = matrix @ camera_point
    return robot_point.tolist()


if __name__ == "__main__":
    transform = save_transform()
    print(transform)
