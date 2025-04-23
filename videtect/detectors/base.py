class VideoDetector:
    def preprocess(self, frame):
        return frame

    def extract_landmarks(self, frame):
        return None

    def analyze(self, frame, landmarks):
        return None

    def visualize(self, frame, landmarks, result):
        return frame

    def process(self, frame):
        frame = self.preprocess(frame)
        landmarks = self.extract_landmarks(frame)
        result = self.analyze(frame, landmarks)
        return self.visualize(frame, landmarks, result)
