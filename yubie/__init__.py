from .main import (
    Topics as Topics,
    VisionModule,
    pubsub
)
from .modules.Vision import vision


def dev():
    print("Initialized Dev Mode")
    vision.main()


def test():
    detector = VisionModule.get_on_demand_detector()
    output = detector.get_detections()
    print(output)
