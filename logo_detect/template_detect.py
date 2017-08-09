# -*- coding: utf-8 -*-

import cv2

from util import template, time_use, test_case


class LogoDector():
    def __init__(self, template_img, logo_shape):
        """
        
        :param template: 识别区域的模板
        :param logo_shape: (logo最左侧与识别点距离，logo最上测与识别点距离，logo宽，logo高)
        """
        self.template_list = self._init_template(template_img, logo_shape)

    def _init_template(self, ori_template, ori_logo_shape):
        """
        生成不同尺寸的模板，以及不同尺寸模板对应的logo_shape
        :param ori_template: 原始模板
        :param ori_logo_shape: 原始模板与logo对应尺寸
        :return: 
        """
        result = []
        width, height = ori_template.shape[::-1]
        for scale in [i * 1.0 / 10 for i in range(1, 20)]:
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
        if max_item[0] > 0.9:
            loc = max_item[1]
            logo_shape = max_item[2]
            # 识别出的区域宽度要在0-width之间，高度亦然
            top_left = (max(loc[0] - logo_shape[0], 0), max(loc[1] - logo_shape[1], 0))
            bottom_right = (min(top_left[0] + logo_shape[2], width), min(top_left[1] + logo_shape[3], height))
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


logo_detector_map = {
    "tencent": LogoDector(template("template_tecent.png"), (25, 25, 330, 100)),
    "xigua": LogoDector(template("template_xigua.jpg"), (40, 20, 130, 140))
}


def detect(img):
    for name, detector in logo_detector_map.iteritems():
        if "xigua" in name:
            location = detector.detect_left(img)
            if location:
                return name,location
        location = detector.detect_right(img)
        if location:
            return name,location
    return None,None


if __name__ == '__main__':
    img = test_case("xigua01.png")
    result = detect(img)
    if result:
        top_left, bottom_right = result
        print result
        from matplotlib import pyplot as plt

        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()