import json
from typing import Optional

import numpy as np

from service.models import ImageConf, WindowBox
from service import handler


def combine_planes(image_path: str, boxes: Optional[str] = None) -> np.array:

    image = ImageConf(path=image_path)    
    if not boxes:
        return handler.convert_image(image)
    
    image.box = [WindowBox(**box) for box in json.loads(boxes)]
    print(image.box )
    handler.make_warps(image)
    return handler.convert_image(image)


