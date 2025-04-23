import cv2
from videtect.config.loader import load_detectors

class MultiDetectorApp:
    def __init__(self, config_path):
        self.detectors = load_detectors(config_path)
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect

            # Process each detector
            for detector in self.detectors:
                frame = detector.process(frame)  # Apply detector-specific processing

            # Display the resulting frame
            cv2.imshow("MultiDetector", frame)

            # Press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
