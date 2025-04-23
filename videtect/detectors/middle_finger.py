from .base import VideoDetector
import cv2
import mediapipe as mp

class MiddleFingerDetector(VideoDetector):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()

    def extract_landmarks(self, frame):
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks
    
    def analyze(self, frame, landmarks_list):
        if not landmarks_list: return None
        for landmarks in landmarks_list:
            finger_tip = landmarks.landmark[8]
            base = landmarks.landmark[5]
            if finger_tip.y < base.y:  # Simplified middle finger check
                return "Middle Finger Detected"
        return None

    def visualize(self, frame, landmarks_list, result):
        if result:
            cv2.putText(frame, result, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
