from .base import VideoDetector
import cv2
import mediapipe as mp
from collections import deque
import time

class SwipeFingerDetector(VideoDetector):
    def __init__(self, max_len=15, swipe_threshold=0.2, cooldown=1.0):
        self.hands = mp.solutions.hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.points_history = deque(maxlen=max_len)
        self.swipe_threshold = swipe_threshold
        self.last_detect_time = 0
        self.cooldown = cooldown
        self.last_handedness = None

    def extract_landmarks(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.last_handedness = results.multi_handedness
        return results.multi_hand_landmarks

    def is_only_index_extended(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended.append(True)
            else:
                extended.append(False)
        return extended == [True, False, False, False]

    def analyze(self, frame, landmarks_list):
        current_time = time.time()
        if not landmarks_list:
            self.points_history.clear()
            return None

        landmarks = landmarks_list[0].landmark

        if not self.is_only_index_extended(landmarks):
            self.points_history.clear()
            return None

        index_tip = landmarks[8]
        self.points_history.append((index_tip.x, index_tip.y))

        if len(self.points_history) >= self.points_history.maxlen:
            x_start, y_start = self.points_history[0]
            x_end, y_end = self.points_history[-1]
            dx = x_end - x_start
            dy = y_end - y_start

            if current_time - self.last_detect_time < self.cooldown:
                return None

            if abs(dx) > abs(dy):
                if abs(dx) > self.swipe_threshold:
                    self.last_detect_time = current_time
                    label = self.get_hand_label()
                    return f"{label} One-Finger Swipe Right" if dx > 0 else f"{label} One-Finger Swipe Left"
            else:
                if abs(dy) > self.swipe_threshold:
                    self.last_detect_time = current_time
                    label = self.get_hand_label()
                    return f"{label} One-Finger Swipe Down" if dy > 0 else f"{label} One-Finger Swipe Up"

        return None

    def get_hand_label(self):
        if self.last_handedness:
            return self.last_handedness[0].classification[0].label  # 'Left' or 'Right'
        return "Unknown Hand"

    def visualize(self, frame, landmarks_list, result):
        if landmarks_list:
            for hand_landmarks in landmarks_list:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
        self.draw_swipe_path(frame)

        if result:
            cv2.putText(frame, result, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        return frame

    def draw_swipe_path(self, frame):
        for i in range(1, len(self.points_history)):
            x1 = int(self.points_history[i-1][0] * frame.shape[1])
            y1 = int(self.points_history[i-1][1] * frame.shape[0])
            x2 = int(self.points_history[i][0] * frame.shape[1])
            y2 = int(self.points_history[i][1] * frame.shape[0])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
