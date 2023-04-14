from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ImageChannels(BaseModel):
    red: int = 0
    green: int = 1
    blue: int = 2


class WindowBox(BaseModel):
    xt: Optional[int] = None
    xb: Optional[int] = None
    yl: Optional[int] = None
    yr: Optional[int] = None


class Warp(BaseModel):
    clean: bool = True
    red: np.ndarray = Field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    green: np.ndarray = Field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    blue: np.ndarray = Field(default_factory=lambda: np.eye(2, 3, dtype=np.float32))
    mode: int = Field(default_factory=lambda: cv2.MOTION_TRANSLATION)

    class Config:
        arbitrary_types_allowed = True


class ImageConf(BaseModel):
    path: str
    channels: ImageChannels = ImageChannels()
    box: list[WindowBox] = [WindowBox()]
    warp: Warp = Warp()

