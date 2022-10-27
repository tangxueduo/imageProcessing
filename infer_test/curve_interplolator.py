import math


class CurveInterpolator:
    def __init__(self, points: list, tension: float):
        self.points = points
        self.tension = tension
        self.arcDivisions = 300

    def get_points(self, step=1, samples=None):
        lengths = self.get_arc_lengths()
        if samples is None:
            samples = lengths[-1] / step

        out = []
        d = 0.0
        while d <= samples:
            u = d / samples
            idx = get_u_to_mapping(u, lengths)
            out.append(self.get_point_at_t(idx, self.points))
            d += 1
        return out

    def get_arc_lengths(self):
        last = self.get_point_at_t(0, self.points)
        lengths = [0]
        sum = 0
        for p in range(1, self.arcDivisions + 1, 1):
            current = self.get_point_at_t(p / self.arcDivisions, self.points)
            sum += vec3_distance(current, last)
            lengths.append(sum)
            last = current
        return lengths

    def get_point_at_t(self, t, points):
        n_points = len(points) - 1
        p = n_points * t
        idx = int(math.floor(p))
        weight = p - idx

        # getControlPoints
        max_index = len(points) - 1
        p0 = points[max(idx - 1, 0)]
        p1 = points[idx]
        p2 = points[min(idx + 1, max_index)]
        p3 = points[min(idx + 2, max_index)]

        target = []
        for i in range(3):
            target.append(solveForT(weight, self.tension, p0[i], p1[i], p2[i], p3[i]))
        return target


def get_u_to_mapping(u, arcLengths):
    il = len(arcLengths)
    targetArcLength = u * arcLengths[il - 1]

    low = 0
    high = il - 1
    while low <= high:
        i = int(math.floor(low + (high - low) / 2))
        comparison = arcLengths[i] - targetArcLength
        if comparison < 0:
            low = i + 1
        elif comparison > 0:
            high = i - 1
        else:
            high = i
            break
    i = high
    if arcLengths[i] == targetArcLength:
        return i / (il - 1)

    lengthBefore = arcLengths[i]
    lengthAfter = arcLengths[i + 1]
    segmentLength = lengthAfter - lengthBefore
    segmentFraction = (targetArcLength - lengthBefore) / segmentLength
    return (i + segmentFraction) / (il - 1)


def solveForT(t, tension, v0, v1, v2, v3):
    EPS = math.pow(2, -42)
    if abs(t) < EPS:
        return v1
    if abs(1 - t) < EPS:
        return v2
    t2 = t * t
    t3 = t * t2
    [a, b, c, d] = getCoefficients(v0, v1, v2, v3, 0, tension)
    return a * t3 + b * t2 + c * t + d


def getCoefficients(v0, v1, v2, v3, v, tension):
    c = (1 - tension) * (v2 - v0) * 0.5
    x = (1 - tension) * (v3 - v1) * 0.5
    a = 2 * v1 - 2 * v2 + c + x
    b = -3 * v1 + 3 * v2 - 2 * c - x
    d = v1 - v
    return [a, b, c, d]


def vec3_distance(a, b):
    x = b[0] - a[0]
    y = b[1] - a[1]
    z = b[2] - a[2]
    return math.sqrt((x * x + y * y + z * z))
