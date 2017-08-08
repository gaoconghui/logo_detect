import os
from functools import wraps

import cv2
import time

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