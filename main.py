from videtect.app.runner import MultiDetectorApp

if __name__ == "__main__":
    app = MultiDetectorApp("videtect/config/config.yaml")
    app.run()
