from yubie.src.abstract_classes import IObjectDetector
from yubie.src.types import ImageObject
from yubie import VisionModule

from ultralytics import YOLO

class YoloV8M(IObjectDetector):
    def __init__(self):
        self.model = YOLO('yolov8m.pt')

    def detect(self, image: ImageObject, classToDetect: list[str]):
        results = self.model.predict(image.data)
        return results


VisionModule.register_model(YoloV8M)
