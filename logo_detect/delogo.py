# -*- coding:utf-8 -*-


"""
清除logo，使用opencv的inpaint
"""

import cv2
import numpy as np

from logo_detect.template_detect import detect


def inpaint(img, logo_location):
    """
    清除logo
    :param img: 原始图像
    :param logo_location: logo位置
    :return: 处理后的图像
    """
    shape = img.shape
    height, width = shape[0], shape[1]

    mask = np.zeros((height, width), np.uint8)
    top_left, bottom_right = logo_location
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


if __name__ == '__main__':
    img = cv2.imread("/Users/gaoconghui/PycharmProjects/logo_detect/logo_detect/test_case/xigua01.png")
    shape = img.shape
    if len(shape) == 3:
        dim = shape[-1]
        if dim == 3:
            img_gary = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif dim == 4:
            img_gary = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError()
    else:
        img_gary = img.copy()
    result = detect(img_gary)
    if result:
        location = result[1]
        dst = inpaint(img, location)
        print cv2.imencode(".jpg", dst)
