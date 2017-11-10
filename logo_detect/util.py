# coding: utf-8
import time
from functools import wraps

import cv2
import os

from logo_detect.settings import TEMPLATE_BASE, TEST_CASE_BASE, TODO_BASE


def template(name):
    return cv2.imread(os.path.join(TEMPLATE_BASE, name), 0)


def template_complete(name):
    return cv2.imread(os.path.join(TEMPLATE_BASE, name), cv2.IMREAD_UNCHANGED)


def test_case(name):
    return cv2.imread(os.path.join(TEST_CASE_BASE, name), 0)


def todo_case(name):
    return cv2.imread(os.path.join(TODO_BASE, name), 0)


def time_use(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print func.__name__ + " time use " + str(t2 - t1)
        return result

    return wrapper


def paint_logo(img, top_left=None, bottom_right=None):
    """
    在原图上勾勒出logo
    :param img: 
    :param top_left: 
    :param bottom_right: 
    :return: 
    """
    from matplotlib import pyplot as plt

    if top_left and bottom_right:
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.imshow(img)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()


def paint_logos(img, locations):
    from matplotlib import pyplot as plt
    for top_left, bottom_right in locations:
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.imshow(img)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    temp = template("duanzi01.png")
    paint_logo(temp)
