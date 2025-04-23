import numpy as np

LEFT = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]

def get_eye_landmarks(landmarks, shape):
    h, w = shape[:2]
    lm = landmarks.landmark
    left = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT])
    right = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT])
    return left, right

def calculate_ear(left, right):
    def ear(eye): 
        return (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2.0 * np.linalg.norm(eye[0] - eye[3]))
    return (ear(left) + ear(right)) / 2
