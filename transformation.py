import cv2
import numpy as np

camera_points = np.array([
    [386,241,820],
    [281,261,830],
    [169,220,830],
    [223,187,724],
    [454,168,690]
], dtype=np.float32)

robot_points = np.array([
    [259,-97,-6],
    [221,46,0.99],
    [249,214,4.35],
    [216,135,110],
    [211,-160,134]
], dtype=np.float32)

returnValue, matrix, inliers2 = cv2.estimateAffine3D(camera_points, robot_points)
print(matrix)
np.save("coord_transform.npy",  matrix)