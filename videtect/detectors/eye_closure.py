from .base import VideoDetector
import cv2
import mediapipe as mp
import numpy as np
from videtect.utils.helpers import get_eye_landmarks, calculate_ear

class EyeClosureDetector(VideoDetector):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    
    def extract_landmarks(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
    
    def analyze(self, frame, landmarks_list):
        if not landmarks_list: return None
        left, right = get_eye_landmarks(landmarks_list[0], frame.shape)
        ear = calculate_ear(left, right)
        return "Closed" if ear < 0.25 else "Open"

    def visualize(self, frame, _, result):
        cv2.putText(frame, f"Eye: {result}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame
