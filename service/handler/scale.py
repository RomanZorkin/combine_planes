import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from service.models import BoxCoord, ScaleConf


def midpoint(pta, ptb) -> tuple[int, int]:
    return (int((pta[0] + ptb[0]) * 0.5), int((pta[1] + ptb[1]) * 0.5))


def get_coord(cnt: np.ndarray) -> BoxCoord:
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = imutils.perspective.order_points(box)
    bx = BoxCoord(box=box)
    tl, tr, br, bl = box
    bx.mid.top.x, bx.mid.top.y = midpoint(tl, tr)
    bx.mid.bottom.x, bx.mid.bottom.y = midpoint(bl, br)
    bx.mid.left.x, bx.mid.left.y = midpoint(tl, bl)
    bx.mid.right.x, bx.mid.right.y = midpoint(tr, br)
    bx.center.x, bx.center.y = bx.mid.top.x, bx.mid.left.y
    bx.height = dist.euclidean((bx.mid.top.x, bx.mid.top.y), (bx.mid.bottom.x, bx.mid.bottom.y))
    bx.width = dist.euclidean((bx.mid.left.x, bx.mid.left.y), (bx.mid.right.x, bx.mid.right.y))
    bx.ratio = bx.width / bx.height
    return bx


def counturs_filter(cnts: tuple[np.ndarray], conf: ScaleConf) -> tuple[np.ndarray, ...]:
    counturs_list = []
    for countur in cnts:
        if cv2.contourArea(countur) < conf.edge_area_min:
            continue
        if cv2.contourArea(countur) > conf.edge_area_max:
            continue
        counturs_list.append(countur)
    return tuple(counturs_list)


def get_contours(full_mask: np.ndarray, conf: ScaleConf) -> tuple[np.ndarray]:
    # нас интересуют только внешние контуры - cv2.RETR_EXTERNAL
    cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = counturs_filter(cnts, conf)
    cnts, _ = imutils.contours.sort_contours(cnts, method='left-to-right')
    return cnts


def lokal_mask(hsv_image: np.ndarray, pallet) -> np.ndarray:
    low = np.array(pallet['low'], dtype='uint8')
    high = np.array(pallet['high'], dtype='uint8')
    return cv2.inRange(hsv_image, low, high)


def get_mask(image: np.ndarray, conf: ScaleConf) -> np.ndarray:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # цикл необходим для получения маски по нескольким диапазанам цвета
    full_mask = np.array([0])
    for pallet in conf.background:
        if full_mask.any():
            full_mask = full_mask + lokal_mask(hsv_img, pallet)
        else:
            full_mask = lokal_mask(hsv_img, pallet)
    # Так как мы передаем политру фона, то инвертируем полученную маску,
    # для выделения объектов на фоне
    full_mask = (full_mask / 255).astype('int')  # noqa:WPS432
    full_mask = np.where((full_mask == 0) | (full_mask == 1), full_mask ^ 1, full_mask)
    full_mask = np.uint8(full_mask)  # type: ignore
    return cv2.GaussianBlur(full_mask, (3, 3), 0)


def axis_ratio(cnts: tuple[np.ndarray]) -> list[float]:
    return [get_coord(cnt).ratio for cnt in cnts]


def get_ratio(image: np.ndarray, conf: ScaleConf) -> float:
    full_mask = get_mask(image, conf)
    cnts = get_contours(full_mask, conf)
    ratio = axis_ratio(cnts)
    return float(np.median(ratio))
