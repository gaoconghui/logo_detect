# coding=utf-8
"""
开发时测试使用
"""
from collections import Counter

import cv2
import numpy as np
import scipy as sp

from logo_detect.util import template, test_case

img1 = template("template_duanzi2.png")  # queryImage
img2 = test_case('duanzi_left02.png')  # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

print 'matches...', len(matches)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.65 * n.distance:
        good.append(m)
print 'good', len(good)
# #####################################
# visualization
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
view[:h1, :w1, 0] = img1
view[:h2, w1:, 0] = img2
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

for m in good:
    # draw the keypoints
    # print m.queryIdx, m.trainIdx, m.distance
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    # print 'kp1,kp2',kp1,kp2
    cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])),
             (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)

match_points = [kp2[m.trainIdx] for m in good]
width, height = img2.shape[::-1]
logo_limit_x, logo_limit_y = width / 2, height / 2

print len(match_points)
print len(set(match_points))


# match_points = [p for p in match_points if p.pt[0] < logo_limit_x and p.pt[1] < logo_limit_y]
# x_center = sum([p.pt[0] for p in match_points]) / len(match_points)
# y_center = sum([p.pt[1] for p in match_points]) / len(match_points)
# width = max([p.pt[0] for p in match_points]) - min([p.pt[0] for p in match_points])
# height = max([p.pt[1] for p in match_points]) - min([p.pt[1] for p in match_points])
#
# color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
# cv2.rectangle(view, (int(x_center + w1 - width / 1.5),int(y_center-height / 1.5)),(int(x_center + w1 + width / 1.5),int(y_center+height / 1.5)), color,1)
#
# detector_map = np.array(img2)
# width,height = detector_map.shape[::-1]
# for point in match_points:
#     x = point.pt[0]
#     y = point.pt[1]

def _remove_dedup_point(points):
    counter = Counter(points)
    return [k for k, v in counter.iteritems() if v == 1]


def _kmeans(points, k):
    _points = np.array([(int(p.pt[0]), int(p.pt[1])) for p in points])
    _points = np.float32(_points)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(_points, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    result = []
    for i in range(k):
        result.append((center[i], _points[label.ravel() == i]))
    return result


match_points = _remove_dedup_point([kp2[m.trainIdx] for m in good])
# 匹配的点可能在很多不同的位置，需要找到一块聚合在一起的区域
k_means_result = _kmeans(match_points, 3)
center, points = max(k_means_result, key=lambda x: len(x[1]))
print points.shape
y_sub = max([p[1] for p in points]) - min([p[1] for p in points])
x_sub = max([p[0] for p in points]) - min([p[1] for p in points])
print x_sub
print y_sub

height = y_sub * 3
width = height * 5

for m in points:
    # draw the keypoints
    # print m.queryIdx, m.trainIdx, m.distance
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    # print 'kp1,kp2',kp1,kp2
    cv2.circle(view, (int(m[0] + w1),int(m[1])),
               3, color)

color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
cv2.rectangle(view, (int(center[0] + w1 - width / 2),int(center[1]-height / 2)),(int(center[0] + w1 + width / 2),int(center[1]+height / 2)), color,1)

cv2.imshow("view", view)
cv2.waitKey()
