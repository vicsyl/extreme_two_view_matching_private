from dataclasses import dataclass
import numpy as np
from typing import List
import cv2 as cv

@dataclass
class ImageData:
    img: np.ndarray
    key_points: List[cv.KeyPoint]
    descriptions: object
    K: np.ndarray
    normals: np.ndarray
    components_indices: np.ndarray
    valid_components_dict: dict
