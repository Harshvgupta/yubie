"""
## Vision Module

This adds the vision functionality to the yubie

"""

from yubie.vision import ObjectDetectionModule, IObjectDetector
from yubie.enumerators import DetectableClasses
from yubie.types import Image, BoundingBox

class YoloV5(IObjectDetector):
    detect(imageData: Image, classToDetect: DetectableClasses) -> BoundingBox:
        pass
    
class TensorFlow(IObjectDetector):
    detect(imageData: Image, classToDetect: DetectableClasses) -> BoundingBox:
        pass

ObjectDetectionModule.register(YoloV5, default = True)
ObjectDetectionModule.register(TensorFlow)