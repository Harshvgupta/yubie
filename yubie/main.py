from typing import Callable, Optional, Dict, Any
from .src.exceptions import TopicNotRegisteredException, ModelNotFoundException
from .src.enums import Topics
from .src.types import StreamingOptions
from .src.abstract_classes import IObjectDetector

# Classes


class PubSubModule():
    def __init__(self, topics: list = []) -> None:
        self._subscribers: Dict[str, list[Callable]] = {}
        for topic in topics:
            self._subscribers[topic] = []

    def publish(self, topic: str, *args: list) -> None:
        if topic not in self._subscribers:
            raise TopicNotRegisteredException(topic)
        for callable in self._subscribers[topic]:
            callable(*args)

    def subscribe_to(self, topic: str, callback: Optional[Callable] = None) -> Any:
        if topic not in self._subscribers:
            raise TopicNotRegisteredException(topic)
        self._subscribers[topic].append(callback)


class OnDemandDetector():
    def __init__(self, model):
        pubsub.subscribe_to(Topics.ImageFromBot, self._store_latest_image)
        self.instance = model()

    def _store_latest_image(self, image):
        self.image = image

    def get_detections(self, classes_to_detect: list[str] = []):
        return self.instance.detect(self.image, classes_to_detect)


class StreamDetector():
    def __init__(self, model, callback):
        pubsub.subscribe_to(Topics.ImageFromBot, self.get_detections)
        self.instance = model()
        self.callback = callback

    def get_detections(self, image):
        if self._publish_detections:
            output = self.instance.detect(image, self.classes_to_detect)
            self.callback(output)

    def start_detections(self, classes_to_detect: list[str]):
        self.classes_to_detect = classes_to_detect
        self._publish_detections = True

    def stop_detections(self):
        self._publish_detections = False


class VisionModule():
    _registered_models = {}
    _instances = {}
    _default_model = None

    @classmethod
    def register_model(cls, model: IObjectDetector, default: bool = False) -> None:
        name = model.__class__.__name__
        cls._registered_models[name] = model
        if default or cls._default_model is None:
            cls._default_model = name

    def get_stream_detector(self, callback: Callable, model_name: str = None, options: StreamingOptions = {}) -> StreamDetector:
        if not model_name:
            model_name = self._default_model

        if model_name not in self._registered_models:
            raise ModelNotFoundException(model_name)

        model = self._registered_models[model_name]
        return StreamDetector(model, callback)

    def get_on_demand_detector(self, model_name: str = None) -> OnDemandDetector:
        if not model_name:
            model_name = self._default_model
            
        if model_name not in self._registered_models:
            raise ModelNotFoundException(model_name)

        model = self._registered_models[model_name]
        return OnDemandDetector(model)


# Main Function
pubsub = PubSubModule(Topics)
vision_module = VisionModule()