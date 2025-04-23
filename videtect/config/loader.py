import yaml
import importlib
import pkgutil
from videtect.detectors.base import VideoDetector

# Auto-discover all detectors in the 'detectors' folder
def discover_detectors():
    detectors = {}
    package = "videtect.detectors"
    for _, name, _ in pkgutil.iter_modules([package.replace(".", "/")]):
        module = importlib.import_module(f"{package}.{name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, VideoDetector) and obj is not VideoDetector:
                detectors[name] = obj
    return detectors

# Load detectors from config file
def load_detectors(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if "active_detectors" not in config:
        raise ValueError("Config must contain 'active_detectors' key.")
    
    detectors = []
    available_detectors = discover_detectors()
    
    for name in config.get("active_detectors", []):
        cls = available_detectors.get(name)
        if cls:
            detectors.append(cls())
        else:
            print(f"Warning: Detector '{name}' not found.")
    return detectors
