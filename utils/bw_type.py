from __future__ import annotations

import copy
import math
from typing import Union


class Line2D:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self) -> str:
        return str(self.__dict__)


class Proposal:
    def __init__(self, bbox: Rectangle, label: str, probability: float):
        self.bbox = copy.deepcopy(bbox)
        self.label = label
        self.probability = probability

    def __repr__(self) -> str:
        return str(self.__dict__)


class Point2D:
    def __init__(self, x, y):
        assert type(x) == type(y)

        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def euclidean_distance(self, p: Point2D) -> float:
        return math.sqrt((self._x - p._x) ** 2 + (self._y - p._y) ** 2)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        return False


class Rectangle:
    def __init__(
        self,
        xmin: Union[int, float],
        ymin: Union[int, float],
        xmax: Union[int, float],
        ymax: Union[int, float],
    ):
        assert xmin <= xmax
        assert ymin <= ymax
        assert type(xmin) == type(ymin) == type(xmax) == type(ymax)

        self._left_top = Point2D(xmin, ymin)
        self._right_bottom = Point2D(xmax, ymax)
        self._center = None
        self._diagonal_distance = None

    @classmethod
    def new_from_point(cls, left_top: Point2D, right_bottom: Point2D):  # type:ignore
        return cls(left_top.x, left_top.y, right_bottom.x, right_bottom.y)

    @property
    def xmin(self):
        return self._left_top.x

    @property
    def ymin(self):
        return self._left_top.y

    @property
    def xmax(self):
        return self._right_bottom.x

    @property
    def ymax(self):
        return self._right_bottom.y

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def center(self):
        if self._center is None:
            if type(self._left_top.x) is int:
                cen_x = (self._left_top.x + self._right_bottom.x) // 2
                cen_y = (self._left_top.y + self._right_bottom.y) // 2
            else:
                cen_x = (self._left_top.x + self._right_bottom.x) / 2.0
                cen_y = (self._left_top.y + self._right_bottom.y) / 2.0
            self._center = Point2D(cen_x, cen_y)

        return copy.deepcopy(self._center)

    @property
    def diagonal_distance(self):
        if self._diagonal_distance is None:
            self._diagonal_distance = self._left_top.euclidean_distance(
                self._right_bottom
            )

        return self._diagonal_distance

    @property
    def area(self):
        if self.height < 0 or self.width < 0:
            return 0
        return self.height * self.width

    def iou(self, rec: Rectangle) -> float:
        xmin = max(self.xmin, rec.xmin)
        ymin = max(self.ymin, rec.ymin)
        xmax = min(self.xmax, rec.xmax)
        ymax = min(self.ymax, rec.ymax)

        if xmin >= xmax or ymin >= ymax:
            return 0.0

        inter = Rectangle(xmin, ymin, xmax, ymax)
        inter_area = inter.area

        union_area = rec.area + self.area - inter.area
        if union_area <= 0:
            return 0.0

        return inter_area / float(union_area)

    def contains(self, rec: Rectangle):
        if (
            self.xmin <= rec.xmin
            and self.ymin <= rec.ymin
            and self.xmax >= rec.xmax
            and self.ymax >= rec.ymax
        ):
            return True

        if (
            rec.xmin <= self.xmin
            and rec.ymin <= self.ymin
            and rec.xmax >= self.xmax
            and rec.ymax >= self.ymax
        ):
            return True

        return False

    def __repr__(self) -> str:
        return str(self.__dict__)


class LineSegment2D:
    def __init__(self, point1: Point2D, point2: Point2D):
        self._point1 = point1
        self._point2 = point2

    @property
    def point1(self) -> Point2D:
        return copy.deepcopy(self._point1)

    @property
    def point2(self) -> Point2D:
        return copy.deepcopy(self._point2)

    @property
    def length(self) -> float:
        return self._point1.euclidean_distance(self._point2)

    def __repr__(self) -> str:
        return str(self.__dict__)


class Diameter2D:
    def __init__(self, line_segment: LineSegment2D, length: float):
        self._line_segment = line_segment
        self._length = length

    @property
    def line_segment(self):
        return copy.deepcopy(self._line_segment)

    @property
    def length(self):
        return self._length

    @property
    def point1(self):
        return self.line_segment.point1

    @property
    def point2(self):
        return self.line_segment.point2

    def __repr__(self) -> str:
        return str(self.__dict__)


class Point3D:
    def __init__(
        self, x: Union[int, float], y: Union[int, float], z: Union[int, float]
    ):
        assert type(x) == type(y) == type(z)

        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def euclidean_distance(self, p: Point3D) -> float:
        return math.sqrt(
            (self._x - p._x) ** 2 + (self._y - p._y) ** 2 + (self._z - p._z) ** 2
        )

    def __repr__(self) -> str:
        return str(self.__dict__)


class Cube:
    def __init__(
        self,
        xmin: Union[int, float],
        ymin: Union[int, float],
        zmin: Union[int, float],
        xmax: Union[int, float],
        ymax: Union[int, float],
        zmax: Union[int, float],
    ):
        assert xmin <= xmax
        assert ymin <= ymax
        assert zmin <= zmax
        assert (
            type(xmin)
            == type(ymin)
            == type(zmin)
            == type(xmax)
            == type(ymax)
            == type(zmax)
        )

        self._left_top_back = Point3D(xmin, ymin, zmin)
        self._right_bottom_front = Point3D(xmax, ymax, zmax)
        self._center = None
        self._diagonal_distance = None

    @classmethod
    def new_from_point(cls, left_top_back: Point3D, right_bottom_front: Point3D):
        return cls(
            left_top_back.x,
            left_top_back.y,
            left_top_back.z,
            right_bottom_front.x,
            right_bottom_front.y,
            right_bottom_front.z,
        )

    @property
    def xmin(self):
        return self._left_top_back.x

    @property
    def ymin(self):
        return self._left_top_back.y

    @property
    def zmin(self):
        return self._left_top_back.z

    @property
    def xmax(self):
        return self._right_bottom_front.x

    @property
    def ymax(self):
        return self._right_bottom_front.y

    @property
    def zmax(self):
        return self._right_bottom_front.z

    @property
    def depth(self):
        return self.zmax - self.zmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def center(self):
        if self._center is None:
            if type(self._left_top_back.x) is int:
                cen_x = (self._left_top_back.x + self._right_bottom_front.x) // 2
                cen_y = (self._left_top_back.y + self._right_bottom_front.y) // 2
                cen_z = (self._left_top_back.z + self._right_bottom_front.z) // 2
            else:
                cen_x = (self._left_top_back.x + self._right_bottom_front.x) / 2.0
                cen_y = (self._left_top_back.y + self._right_bottom_front.y) / 2.0
                cen_z = (self._left_top_back.z + self._right_bottom_front.z) / 2.0
            self._center = Point3D(cen_x, cen_y, cen_z)

        return copy.deepcopy(self._center)

    @property
    def diagonal_distance(self):
        if self._diagonal_distance is None:
            self._diagonal_distance = self._left_top_back.euclidean_distance(
                self._right_bottom_front
            )

        return self._diagonal_distance

    def __repr__(self) -> str:
        return str(self.__dict__)
