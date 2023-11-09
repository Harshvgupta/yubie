from yubie.src.abstract_classes import IObjectDetector
from yubie.src.types import ImageObject
from yubie import VisionModule

from ultralytics import YOLO
import json


class YoloV8M(IObjectDetector):
    def __init__(self):
        self.model = YOLO(
            '/Users/pranomvignesh/Workfolder/yubie/yubie/models/vision/yolov8m.pt')

    def detect(self, image: ImageObject, classToDetect: list[str] = []):
        results = self.model.predict(
            image.data, verbose=False, classes=[0, 41])
        return list(map(self.postprocess, results))

    def render(self, image, classToDetect: list[str] = [0, 41]):
        results = self.model.predict(
            image, show=True, verbose=False, classes=classToDetect)
        # print(results)
        # output = results.render()
        # return output[0]

    def postprocess(self, result):
        json_string = result.tojson()
        return json.loads(json_string)


VisionModule.register_model(YoloV8M)
