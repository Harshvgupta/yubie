from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(kw_only=True)
class ImageObject:
    label: str
    data: np.ndarray
    meta: Dict[str, any] = {}


@dataclass(kw_only=True)
class StreamingOptions():
    calls_per_second: float
    resumable: bool = False
