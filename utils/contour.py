import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .bw_math import round_half_up
from .bw_type import Diameter2D, LineSegment2D, Point2D, Rectangle

np_round = np.vectorize(round_half_up)


class _Line:
    def __init__(self, intercept: np.float32, slope: np.float32):
        self.intercept = intercept
        self.slope = slope


class CVContour:
    def __init__(self, points, spacing=(1.0, 1.0)):
        self.spacing = spacing
        self._points = points
        self._long_diameter: Optional[Diameter2D] = None
        self._short_diameter: Optional[Diameter2D] = None
        self._bbox = None

    @property
    def points(self):
        return self._points

    @property
    def short_diameter(self):
        if self._short_diameter is None:
            self._long_diameter, self._short_diameter = get_diameter(
                self.points, self.spacing
            )
        return self._short_diameter

    @property
    def long_diameter(self):
        if self._long_diameter is None:
            self._long_diameter, self._short_diameter = get_diameter(
                self.points, self.spacing
            )
        return self._long_diameter

    @property
    def bbox(self):
        if self._bbox is None:
            x, y, w, h = cv2.boundingRect(self.points)
            self._bbox = Rectangle(xmin=x, ymin=y, xmax=x + w, ymax=y + h)
        return self._bbox

    def bbox_iou(self, other):
        x1, y1, w1, h1 = cv2.boundingRect(self.points)
        x2, y2, w2, h2 = cv2.boundingRect(other.points)
        reg1 = Rectangle(xmin=x1, ymin=y1, xmax=x1 + w1, ymax=y1 + h1)
        reg2 = Rectangle(xmin=x2, ymin=y2, xmax=x2 + w2, ymax=y2 + h2)

        return reg1.iou(reg2)

    def merge(self, *others):
        for cnt in others:
            if cnt.spacing != self.spacing:
                raise Exception(
                    f"can not merge two contour with different spacing {cnt.spacing} vs {self.spacing}"
                )

        white, fill, draw_all, padding = 1, -1, -1, 10
        points_arr = np.concatenate([self.points] + [cnt.points for cnt in others])

        height, width = (
            np.max(points_arr[:, :, 0]) + padding,
            np.max(points_arr[:, :, 1]) + padding,
        )
        canvas_mat = np.zeros((height, width), dtype="uint8")

        for points in points_arr:
            canvas_mat = cv2.drawContours(canvas_mat, [points], draw_all, white, fill)
        merged_points = _find_max_contour(
            canvas_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE, 0, 0
        )

        return CVContour(merged_points, self.spacing)

    def __gt__(self, other):
        if not isinstance(other, CVContour):
            return False

        return float(self.short_diameter.length) > float(other.short_diameter.length)


def get_diameter(
    cnt_points: np.ndarray, spacing_xy: Tuple[float, float]
) -> Tuple[Diameter2D, Diameter2D]:
    long = _get_long_diameter(cnt_points, spacing_xy)

    short = _get_short_diameter(long, cnt_points, spacing_xy)

    return long, short


def _get_short_diameter(
    long: Diameter2D, cnt_points: np.ndarray, spacingxy: Tuple[float, float]
) -> Diameter2D:
    class IndexDist:
        def __init__(self, index: int, distance: float):
            self.index = index
            self.distance = distance

    class IndexPoint:
        def __init__(self, index: int, p: Point2D):
            self.index = index
            self.point = p

    ld_line = _calc_line(long.line_segment.point1, long.line_segment.point2)

    start_p, end_p = long.line_segment.point1, long.line_segment.point2

    if start_p.x != end_p.x:
        if end_p.x - start_p.x < 0:
            start_p, end_p = end_p, start_p
    else:
        if end_p.y - start_p.y < 0:
            start_p, end_p = end_p, start_p

    points: List[Point2D] = []

    project_point_dists: List[IndexDist] = []

    for i, p in enumerate(cnt_points):
        point = Point2D(p[0], p[1])
        points.append(point)
        pp = _project_point_onto_line(point, ld_line)
        dist = _get_pixel_distance(pp, start_p)
        project_point_dists.append(IndexDist(i, dist))

    project_point_dists = sorted(project_point_dists, key=lambda x: x.distance)

    if ld_line.intercept == 0:
        sd_line = _Line(np.float32(-1 / 1e-5), np.float32(0))
    else:
        sd_line = _Line(-1 / ld_line.intercept, np.float32(0))

    j = 0
    sp1, sp2 = Point2D(0, 0), Point2D(0, 0)
    short_diameter = 0.0
    pixel_dist = _get_pixel_distance(long.line_segment.point1, long.line_segment.point2)
    for i in range(int(pixel_dist)):
        pps: List[IndexPoint] = []
        for d in project_point_dists[j:]:
            if d.distance > i + 0.5:
                j = i
                break

            if d.distance > i - 0.5:
                pp = _project_point_onto_line(points[d.index], sd_line)
                pps.append(IndexPoint(d.index, pp))

        if len(pps) < 2:
            continue

        if sd_line.intercept == 0:
            pps = sorted(pps, key=lambda x: x.point.x)
        else:
            pps = sorted(pps, key=lambda x: x.point.y)

        p1 = points[pps[0].index]
        p2 = points[pps[-1].index]
        dist = _get_pixel_distance(p1, p2)
        if dist > short_diameter:
            short_diameter = dist
            sp1, sp2 = p1, p2

    length = _get_physical_distance(sp1, sp2, spacingxy)

    short = Diameter2D(LineSegment2D(sp1, sp2), length)

    return short


def _find_max_contour(src_mat: np.ndarray, mode, method, offset_x: int, offset_y: int):
    contours, _ = cv2.findContours(src_mat, mode, method)
    if len(contours) < 1:
        return None
    max_points = max(contours, key=cv2.contourArea)
    if len(max_points) == 0:  # 14*1*2
        return None
    if offset_x != 0 or offset_y != 0:
        max_points[:, :, 0] += offset_x
        max_points[:, :, 1] += offset_y
    return max_points  # 14*1*2


def _find_polar_pair(contour):
    rect = cv2.minAreaRect(contour)
    brect = np_round(cv2.boxPoints(rect)).astype(np.int)

    min_dist = -1.0
    closest_idx = 0
    others = []
    p1 = Point2D(brect[0][0], brect[0][1])
    for i, p in enumerate(brect):
        if i == 0:
            continue

        p2 = Point2D(brect[i][0], brect[i][1])
        dist = _get_pixel_distance(p1, p2)

        if min_dist < 0 or dist < min_dist:
            min_dist = dist
            if closest_idx > 0:
                others.append(closest_idx)
            closest_idx = i
        else:
            others.append(i)
    p2 = Point2D(brect[closest_idx][0], brect[closest_idx][1])
    l1 = _calc_line(p1, p2)

    p1 = Point2D(brect[others[0]][0], brect[others[0]][1])
    p2 = Point2D(brect[others[1]][0], brect[others[1]][1])
    l2 = _calc_line(p1, p2)
    return l1, l2


def _get_physical_distance(
    p1: Point2D, p2: Point2D, spacing_xy: Tuple[float, float]
) -> float:
    dx = (p1.x - p2.x) * spacing_xy[0]
    dy = (p1.y - p2.y) * spacing_xy[1]
    return math.hypot(dx, dy)


def _get_pixel_distance(p1: Point2D, p2: Point2D) -> float:
    return math.hypot(float(p1.x - p2.x), float(p1.y - p2.y))


def _get_long_diameter(
    cnt_points: np.ndarray, spacingxy: Tuple[float, float]
) -> Diameter2D:
    hull = cv2.convexHull(cnt_points, clockwise=True)
    l1, l2 = _find_polar_pair(hull)

    min_dist1, min_dist2 = -1.0, -1.0
    p1, p2 = Point2D(0, 0), Point2D(0, 0)
    for p in hull:
        x, y = np.float32(p[0][0]), np.float32(p[0][1])
        dist1 = np.abs(l1.intercept * x + l1.slope - y)
        dist2 = np.abs(l2.intercept * x + l2.slope - y)

        if min_dist1 < 0 or dist1 < min_dist1:
            p1 = Point2D(x, y)
            min_dist1 = dist1

        if min_dist2 < 0 or dist2 < min_dist2:
            p2 = Point2D(x, y)
            min_dist2 = dist2
    length = _get_physical_distance(p1, p2, spacingxy)
    return Diameter2D(LineSegment2D(p1, p2), length)


def _calc_line(p1, p2) -> _Line:
    k = np.float32(p1.y - p2.y) / (np.float32(p1.x - p2.x) + np.float32(1e-5))
    b = np.float32(p1.y) - k * np.float32(p1.x)
    return _Line(k, b)


def _project_point_onto_line(p: Point2D, lab: _Line) -> Point2D:
    x = (lab.intercept * np.float32(p.y - lab.slope) + np.float32(p.x)) / (
        lab.intercept * lab.intercept + np.float32(1)
    )
    y = lab.intercept * x + lab.slope
    return Point2D(int(x), int(y))
