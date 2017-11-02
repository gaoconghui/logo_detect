# -*- coding:utf-8 -*-


"""
清除logo，使用opencv的inpaint
"""

import cv2
import numpy as np

from logo_detect.template_detect import detect


def inpaint(img, logo_location):
    """
    清除logo。 这个有个问题，之前delogo得到的区域不会让值到边缘0，而此处是需要的，所以需要对logo_location做一些预处理
    :param img: 原始图像
    :param logo_location: logo位置
    :return: 处理后的图像
    """
    shape = img.shape
    height, width = shape[0], shape[1]

    mask = np.zeros((height, width), np.uint8)
    top_left, bottom_right = logo_location
    top_left_x, top_left_y = top_left
    bottom_right_x, bottom_right_y = bottom_right
    cv2.rectangle(mask, (top_left_x, top_left_y - 1), (bottom_right_x + 1, bottom_right_y), 255, -1)
    dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return dst


if __name__ == '__main__':
    img = cv2.imread("/Users/gaoconghui/PycharmProjects/logo_detect/logo_detect/test_case/xigua_part01.jpeg")
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
    result = detect(img_gary, strict=1)
    if result:
        location = result[1]
        dst = inpaint(img, location)
        print cv2.imencode(".jpg", dst)
        from matplotlib import pyplot as plt

        plt.imshow(dst)
        plt.show()
