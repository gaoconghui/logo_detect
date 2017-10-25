# -*- coding: utf-8 -*-

from collections import OrderedDict

import cv2

from util import template, time_use, test_case, paint_logo


class LogoDector():
    def __init__(self, template_img, logo_shape, threshold):
        """
        
        :param template: 识别区域的模板
        :param logo_shape: (logo最左侧与识别点距离，logo最上测与识别点距离，logo宽，logo高)
        """
        self.template_list = self._init_template(template_img, logo_shape)
        self.threshold = threshold

    def _init_template(self, ori_template, ori_logo_shape):
        """
        生成不同尺寸的模板，以及不同尺寸模板对应的logo_shape
        :param ori_template: 原始模板
        :param ori_logo_shape: 原始模板与logo对应尺寸
        :return: 
        """
        result = []
        width, height = ori_template.shape[::-1]
        for scale in [i * 1.0 / 10 for i in range(3, 20)]:
            size = (int(width * scale), int(height * scale))
            scale_img = cv2.resize(ori_template, size)
            result.append((
                scale_img,
                tuple([int(i * scale) for i in ori_logo_shape])
            ))
        return result

    @time_use
    def detect(self, img_detect):
        height, width = img_detect.shape
        max_val_loc = []
        for template, logo_shape in self.template_list:
            template_height, template_width = template.shape
            if template_height > height or template_width > width:
                continue
            img = img_detect.copy()
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            max_val_loc.append((max_val, max_loc, logo_shape))
        max_item = max(max_val_loc, key=lambda x: x[0])
        print max_item
        if max_item[0] > self.threshold:
            loc = max_item[1]
            logo_shape = max_item[2]
            # 识别出的区域宽度要在0-width之间，高度亦然
            top_left = (max(loc[0] - logo_shape[0], 1), max(loc[1] - logo_shape[1], 1))
            bottom_right = (min(top_left[0] + logo_shape[2], width - 1), min(top_left[1] + logo_shape[3], height - 1))
            return top_left, bottom_right
        else:
            return None

    @time_use
    def detect_right(self, img_detect):
        width, height = img_detect.shape[::-1]
        crop_left_bottom = (int(width * 1.0 / 5 * 4), int(height * 1.0 / 5))
        crop_img = img_detect[0:crop_left_bottom[1], crop_left_bottom[0]:width]
        logo_location = self.detect(crop_img)
        if logo_location:
            top_left, bottom_right = logo_location
            return (top_left[0] + crop_left_bottom[0], top_left[1]), (
                bottom_right[0] + crop_left_bottom[0], bottom_right[1])
        return None

    @time_use
    def detect_left(self, img_detect):
        width, height = img_detect.shape[::-1]
        crop_right_bottom = (int(width * 1.0 / 4), int(height * 1.0 / 4))
        crop_img = img_detect[0:crop_right_bottom[1], 0:crop_right_bottom[0]]
        logo_location = self.detect(crop_img)
        if logo_location:
            return logo_location
        return None


# to keep it order
logo_detector_map = OrderedDict()
logo_detector_map['xigua_right'] = LogoDector(template("template_xigua.jpg"), (40, 20, 130, 140), 0.85)
logo_detector_map['xigua_left'] = LogoDector(template("template_xigua.jpg"), (40, 20, 130, 140), 0.85)
logo_detector_map['tencent_right'] = LogoDector(template("template_tecent.png"), (25, 25, 330, 100), 0.92)
logo_detector_map['btime_left'] = LogoDector(template("template_btime.png"), (20, 20, 100, 100), 0.9)


def detect(img):
    result = None
    for name, detector in logo_detector_map.iteritems():
        if "left" in name:
            result = detector.detect_left(img)
        if "right" in name:
            result = detector.detect_right(img)
        if result:
            return name, result
    return None, None


# 某些很大的图经常会匹配到很小的logo模板，暂时用不到
def resize_and_detect(img):
    # resize
    width, height = img.shape[::-1]
    scale = 1
    if width > 1200 or height > 1200:
        scale = 0.5
    if scale < 1:
        size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, size)
    name, location = detect(img)
    print name, location
    # restore
    if scale < 1 and location:
        location = tuple([tuple([int(size * scale) for size in shape]) for shape in location])
    return name, location


if __name__ == '__main__':
    detector = logo_detector_map['btime_left']
    img = test_case("btime02.png")
    print img.shape
    name = "btime"
    result = detector.detect_left(img)
    print name, result
    if result:
        top_left, bottom_right = result
        print result
        paint_logo(img, top_left, bottom_right)
