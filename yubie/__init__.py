from .main import (
    Topics as Topics,
    vision_module as VisionModule,
    pubsub
)
from .modules.Vision import vision
from .test import vision_test


def dev():
    # Here multiple modules can be called
    print("Initialized Dev Mode")


def test():
    # Here multiple test files can be called and tested
    vision_test.main()
