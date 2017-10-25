# coding: utf-8
import os
import time
from functools import wraps

import cv2

from logo_detect.settings import TEMPLATE_BASE, TEST_CASE_BASE


def template(name):
    return cv2.imread(os.path.join(TEMPLATE_BASE, name), 0)


def test_case(name):
    return cv2.imread(os.path.join(TEST_CASE_BASE, name), 0)


def time_use(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print "time use " + str(t2 - t1)
        return result

    return wrapper


def paint_logo(img, top_left, bottom_right):
    """
    在原图上勾勒出logo
    :param img: 
    :param top_left: 
    :param bottom_right: 
    :return: 
    """
    from matplotlib import pyplot as plt

    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
