import json
from json import JSONEncoder
from typing import Optional

import cv2
import numpy as np

from service.models import ImageConf, Warp, WindowBox
from service import handler


def combine_planes(image_conf: ImageConf, boxes: Optional[str] = None) -> tuple[np.array, ImageConf]:
    image = cv2.cvtColor(cv2.imread(image_conf.path), cv2.COLOR_BGR2RGB)

    if not boxes:        
        return handler.convert_image(image, image_conf), image_conf
    
    image_conf.box = [WindowBox(**box) for box in json.loads(boxes)]
    handler.make_warps(image, image_conf)
    return handler.convert_image(image, image_conf), image_conf
