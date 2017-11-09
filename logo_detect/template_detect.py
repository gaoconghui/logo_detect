# -*- coding: utf-8 -*-

from collections import OrderedDict, Counter

import cv2
import numpy as np

from util import template, time_use, test_case, paint_logo


class LogoDector():
    @time_use
    def detect(self, img_detect):
        raise ValueError("error")

    @time_use
    def detect_right(self, img_detect):
        width, height = img_detect.shape[::-1]
        crop_left_bottom = (int(width * 1.0 / 5 * 3), int(height * 1.0 / 5))
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


class LogoDectorMatch(LogoDector):
    def __init__(self, template_img, logo_shape, threshold):
        """

        :param template: 识别区域的模板
        :param logo_shape: (logo最左侧与识别点距离，logo最上测与识别点距离，logo宽，logo高)
        """
        self.template_list = self.init_template(template_img, logo_shape)
        self.threshold = threshold

    def init_template(self, ori_template, ori_logo_shape):
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
        if max_item[0] > self.threshold:
            loc = max_item[1]
            logo_shape = max_item[2]
            # 识别出的区域宽度要在0-width之间，高度亦然
            top_left = (max(loc[0] - logo_shape[0], 1), max(loc[1] - logo_shape[1], 1))
            bottom_right = (
                min(top_left[0] + logo_shape[2], width - 1), min(top_left[1] + logo_shape[3], height - 1))
            return top_left, bottom_right
        else:
            return None


class LogoDectorSift(LogoDector):
    def __init__(self, template_img, ratio, threshold, center_location=(0.5, 0.5, 0.5, 0.5)):
        """
        
        :param template_img: 
        :param ratio: 长宽比
        :param threshold: 匹配的点个数
        :param center_location: 匹配出中心点距离四周的位置，分别问上下左右
        """
        self.threshold = threshold
        self.template_img = template_img
        self.ratio = ratio
        self.sift = cv2.SIFT()
        _, self.template_des = self.sift.detectAndCompute(self.template_img, None)
        self.center_location = center_location

    @time_use
    def detect(self, img_detect):

        scale, img_detect = self._resize(img_detect)

        kp2, des2 = self.sift.detectAndCompute(img_detect, None)
        if not kp2 and not des2:
            return None
        try:
            match_points = self._find_good_points(kp2, des2)
        except:
            return None
        if len(match_points) < self.threshold:
            return None
        # 匹配的点可能在很多不同的位置，需要找到一块聚合在一起的区域
        center, points = self._maybe_correct_region(match_points)

        # 通过可能区域，估算长宽比为ratio的长方形logo区域
        # 假设高度为高度差的4倍
        x_list = [p[0] for p in points]
        y_list = [p[1] for p in points]
        y_sub = max(y_list) - min(y_list)
        logo_height = y_sub * 4
        min_logo_width = max(x_list) - min(x_list)
        logo_width = max(logo_height * self.ratio, min_logo_width)
        img_width, img_height = img_detect.shape[::-1]

        # center = (int(max(x_list) + min(x_list)) / 2, int(max(y_list) + min(y_list)) / 2)

        # 还原大小
        logo_width, logo_height, img_width, img_height = (
            int(logo_width / scale), int(logo_height / scale), int(img_width / scale), int(img_height / scale))

        # 计算
        top_left = (int(max(center[0] - logo_width * self.center_location[2], 1)),
                    int(max(center[1] - logo_height * self.center_location[0], 1)))
        bottom_right = (
            int(min(center[0] + logo_width * self.center_location[3], img_width - 1)),
            int(min(center[1] + logo_height * self.center_location[1], img_height - 1)))
        return top_left, bottom_right

    def _resize(self, img):
        width, height = img.shape[::-1]
        scale = 720.0 / max(width, height)
        if scale >= 1:
            return 1, img
        return scale, cv2.resize(img, (int(width * scale), int(height * scale)))

    def _find_good_points(self, point, des):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        print self.template_des.shape
        print des.shape
        matches = flann.knnMatch(self.template_des, des, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)
        match_points = self._remove_dedup_point([point[m.trainIdx] for m in good])
        return match_points

    def _remove_dedup_point(self, points):
        counter = Counter(points)
        return [k for k, v in counter.iteritems() if v == 1]

    def _maybe_correct_region(self, match_points):
        # 假设有三堆，先分个类
        k_means_result = self._kmeans(match_points, 3)
        k_means_result = sorted(k_means_result, key=lambda x: -len(x[1]))
        center, points = k_means_result[0]

        x_list = [p[0] for p in points]
        y_list = [p[1] for p in points]
        x_sub = max(x_list) - min(x_list)
        y_sub = max(y_list) - min(y_list)
        for center_, points_ in k_means_result[1:]:
            if abs(center[0] - center_[0]) < x_sub * 3 and abs(center[1] - abs(center_[1])) < y_sub:
                center = (center[0] + center_[0]) / 2, (center[1] + center_[1]) / 2
                points = np.vstack((points_, points))
        # center, points = max(k_means_result, key=lambda x: len(x[1]))
        return center, points

    def _kmeans(self, points, k):
        _points = np.array([(int(p.pt[0]), int(p.pt[1])) for p in points])
        _points = np.float32(_points)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(_points, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        result = []
        for i in range(k):
            result.append((center[i], _points[label.ravel() == i]))
        return result


# to keep it order
logo_detector_map = OrderedDict()
logo_detector_map['xigua_right'] = LogoDectorMatch(template("template_xigua.jpg"), (40, 20, 130, 140), 0.85)
logo_detector_map['xigua_left'] = LogoDectorMatch(template("template_xigua.jpg"), (40, 20, 130, 140), 0.85)
logo_detector_map['duanzi_left'] = LogoDectorSift(template("template_duanzi.png"), ratio=5, threshold=15,
                                                  center_location=(0.5, 0.5, 0.6, 0.4))
logo_detector_map['tencent_right'] = LogoDectorMatch(template("template_tecent.png"), (25, 25, 330, 100), 0.92)

logo_detector_strict_map = OrderedDict()
logo_detector_strict_map['xigua_right'] = LogoDectorMatch(template("template_xigua.jpg"), (40, 20, 130, 140), 0.80)
logo_detector_strict_map['xigua_left'] = LogoDectorMatch(template("template_xigua.jpg"), (40, 20, 130, 140), 0.80)
logo_detector_strict_map['xigua_part_right'] = LogoDectorMatch(template("template_xigua_part.png"), (60, 90, 220, 270),
                                                               0.80)
logo_detector_strict_map['tencent_right'] = LogoDectorMatch(template("template_tecent.png"), (25, 25, 330, 100), 0.92)
logo_detector_map['duanzi_left'] = LogoDectorSift(template("template_duanzi.png"), ratio=5, threshold=15,
                                                  center_location=(0.5, 0.5, 0.6, 0.4))


# logo_detector_map['btime_left'] = LogoDector(template("template_btime.png"), (20, 20, 100, 100), 0.9)
# logo_detector_map['btime2_left'] = LogoDector(template("template_btime2.png"), (12, 12, 200, 50), 0.9)
# logo_detector_map['btime2_right'] = LogoDector(template("template_btime2.png"), (12, 12, 200, 50), 0.9)


def detect(img, strict=False):
    """
    识别logo区域
    :param img: 
    :param strict: 是否宁可错杀不可放过
    :return: 
    """
    result = None
    detector_map = logo_detector_strict_map if strict else logo_detector_map
    for name, detector in detector_map.iteritems():
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

    detector = logo_detector_map['duanzi_left']
    img = test_case("duanzi_left02.png")
    print img.shape
    name = "btime"
    result = detector.detect_left(img)
    print name, result
    if result:
        top_left, bottom_right = result
        print result
        paint_logo(img, top_left, bottom_right)


        # temp = cv2.imread(os.path.join(TEMPLATE_BASE, "template_duanzi.png"),cv2.IMREAD_UNCHANGED)
        # # channels = cv2.split(temp)
        # # zero_channel = np.zeros_like(channels[0])
        # # mask = np.array(channels[0])
        # # print channels[0]
        # # mask[channels[0] < 100] = 1
        # # mask[channels[0] >= 100] = 0
        # # print mask
        # cv2.imwrite("/tmp/result.png",temp)
        # channels = cv2.split(temp)
        # print len(channels)
        # paint_logo(temp)
        # # transparent_mask = cv2.merge([zero_channel, zero_channel, zero_channel, mask])
