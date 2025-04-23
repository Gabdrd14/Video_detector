from .base import VideoDetector
import cv2
from fer import FER

class EmotionDetector(VideoDetector):
    def __init__(self):
        self.detector = FER()

    def analyze(self, frame, landmarks_list):
        # Get the top emotion from the FER detector
        emotion, score = self.detector.top_emotion(frame)
        return emotion, score

    def visualize(self, frame, _, result):
        # Display the emotion and its confidence score
        emotion, score = result
        cv2.putText(frame, f"{emotion}: {score:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame
