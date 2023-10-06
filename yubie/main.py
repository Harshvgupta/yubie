from enum import Enum
from dataclasses import dataclass


class DetectableClasses(Enum):
    BALL = 1
    PERSON = 2
    
class ImageLabels(Enum):
    FRONTLEFT_FISHEYE_IMAGE = 1
    FRONTRIGHT_FISHEYE_IMAGE = 2
    LEFT_FISHEYE_IMAGE = 3
    RIGHT_FISHEYE_IMAGE = 4
    UNITREE_IMAGE = 5

class ImageSources(Enum):
    TRAINING = 1
    SPOT_LIVE = 2
    UNITREE_LIVE = 3
@dataclass
class Image:
    data: ImageBuffer
    label: ImageLabels
    src: ImageSources
    
@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float
    

def init():
    pass

def benchmark():
    pass

def test():
    pass
