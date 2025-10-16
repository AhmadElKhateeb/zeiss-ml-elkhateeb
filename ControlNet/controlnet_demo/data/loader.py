from dataclasses import dataclass
from typing import Tuple
import numpy as np
import imageio

from annotator.util import resize_image, HWC3

@dataclass
class ImageSpec:
    path: str
    image_resolution: int

class ImageLoader:
    def __init__(self, spec: ImageSpec):
        self.spec = spec

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        img = imageio.imread(self.spec.path)
        img = HWC3(img)
        resized = resize_image(img, self.spec.image_resolution)
        return img, resized