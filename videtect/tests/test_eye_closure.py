from videtect.detectors.eye_closure import EyeClosureDetector

def test_eye_logic_threshold():
    detector = EyeClosureDetector()
    assert detector._ear([[0,1],[0,1],[0,1],[0,2],[0,1],[0,1]], [[0,1],[0,1],[0,1],[0,2],[0,1],[0,1]]) < 0.25
