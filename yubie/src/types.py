from dataclasses import dataclass
from typing import Dict
from PIL import Image
import numpy as np


@dataclass
class ImageObject:
    label: str
    data: np.ndarray | Image.Image


@dataclass
class StreamingOptions():
    calls_per_second: float
    resumable: bool = False
