from setuptools import setup, find_packages

setup(
    name='videtect',
    version='0.1',
    packages=find_packages(),
    install_requires=['opencv-python', 'mediapipe', 'numpy', 'PyYAML'],
)
