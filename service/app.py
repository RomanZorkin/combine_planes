import json
from typing import Optional

import cv2
import numpy as np
from service.handler import planes, scale
from service.models import ImageConf, ScaleConf, WindowBox


def combine_planes(
    image_conf: ImageConf, boxes: Optional[str] = None,
) -> tuple[np.ndarray, ImageConf]:
    image = cv2.cvtColor(cv2.imread(image_conf.path), cv2.COLOR_BGR2RGB)

    if not boxes:
        return planes.convert(image, image_conf), image_conf

    image_conf.box = [WindowBox(**box) for box in json.loads(boxes)]
    planes.make_warps(image, image_conf)
    return planes.convert(image, image_conf), image_conf


def convert_image(image: np.ndarray, conf: ImageConf) -> np.ndarray:
    return planes.convert(image, conf)


def stretching_count(image: np.ndarray, conf: ScaleConf) -> float:
    return scale.get_ratio(image, conf)


def scale_correction(image: np.ndarray, scale_coeff: float) -> np.ndarray:
    height, width, _ = image.shape
    new_height = int(height * scale_coeff)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
