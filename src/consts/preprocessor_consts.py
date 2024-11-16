import cv2

ROTATIONS = [0, cv2.ROTATE_90_CLOCKWISE,
             cv2.ROTATE_180,
             cv2.ROTATE_90_COUNTERCLOCKWISE]
ANGLES = [0, 90, 180, 270]
FIRST_MODEL_PATH = "src/weights/checkpoint_gcnet.pkl"
SECOND_MODEL_PATH = "src/weights/checkpoint_drnet.pkl"
