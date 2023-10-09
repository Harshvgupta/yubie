from abc import ABC, abstractmethod

from .types import ImageObject

class IObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: ImageObject, classToDetect: list[str]):
        pass
