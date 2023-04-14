import cv2
import matplotlib.pyplot as plt
import numpy as np

from service.models import ImageConf


def warp_mean(warps: list[list[np.array], list[np.array]], conf: ImageConf) -> None:
    """Функция усреднения матриц смещения.
    
    После обновления матриц признак чистого класса матриц смещения меняется на - неверно.
    """
    conf.warp.red = np.mean( np.array(warps[0]), axis=0)
    conf.warp.blue = np.mean( np.array(warps[1]), axis=0)
    conf.warp.clean = False


def shift_estimation(image_arr: np.array, conf: ImageConf, num: int) -> tuple[np.array, np.array]:
    """Функция расчета матриц смещения.
    
    Функция производит расчет смещения цветовых каналов от зеленогою в выбранном участке\n
    изображения.
    """
    box = conf.box[num]    
    chunk_img = image_arr[box.xt:box.xb, box.yl:box.yr, :].copy()
    chunk_img_r = chunk_img[:,:,conf.channels.red].copy()
    chunk_img_g = chunk_img[:,:,conf.channels.green].copy()
    chunk_img_b = chunk_img[:,:,conf.channels.blue].copy()

    _, warp_matrix_r = cv2.findTransformECC(chunk_img_g, chunk_img_r, conf.warp.red, conf.warp.mode)
    _, warp_matrix_b = cv2.findTransformECC(chunk_img_g, chunk_img_b, conf.warp.blue, conf.warp.mode)

    return warp_matrix_r, warp_matrix_b


def make_warps(image: np.ndarray, conf: ImageConf) -> None:
    """Функция расчета матриц смещения по всем заданным участкам изображения."""
    
    warps = [[],[]]
    for num in range(len(conf.box)):
        warp_matrix_r, warp_matrix_b  = shift_estimation(image, conf, num)
        warps[0].append(warp_matrix_r)
        warps[1].append(warp_matrix_b)
    warp_mean(warps, conf)    


def convert_image(image: np.array, conf: ImageConf) -> np.array:
    """Функция преобразования изображения на основе матриц смещения цветовых каналов."""
   
    height, width , _ = image.shape

    new_img = np.zeros(image.shape, dtype=np.uint8)

    new_img[:,:,conf.channels.green] = image[:,:,conf.channels.green]
    new_img[:,:,conf.channels.red] = cv2.warpAffine(
        image[:,:,conf.channels.red],
        conf.warp.red,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    new_img[:,:,conf.channels.blue] = cv2.warpAffine(
        image[:,:,conf.channels.blue],
        conf.warp.blue,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )    
    return new_img
