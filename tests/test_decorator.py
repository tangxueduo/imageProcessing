# -*- coding: utf-8 -*-
import functools
import time
from functools import reduce

from loguru import logger

"""装饰器：利用函数作为对象被引用的特点，动态增加函数功能的一种方法。
"""


def timer(func):
    """实现一个时间统计装饰器"""
    t0 = time.time()

    @functools.wraps(func)
    def wrapper(*args, **kw):
        f = func(*args, **kw)
        logger.info(f"call {func.__name__}, cost: {time.time() - t0}")
        return f

    return wrapper


@timer
def test_decorator():
    x, y = 11, 22
    logger.info(f"x:{x}, y{y}")
    return x + y


"""functools.partial, 实现偏函数。
把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
"""


def test_partial():
    max2 = functools.partial(max, 10)
    logger.info(max2(3, 4))


def test_map_reduce():
    """
    用map和reduce编写一个str2float函数，把字符串'123.456'转换成浮点数123.456
    map(fn, iter)
    reduce(fn, iter)
    """

    def str2float(s):
        point_idx = s.find(".")
        if point_idx != -1:
            pow = len(s) - 1 - point_idx
            s = s[:point_idx] + s[point_idx + 1 :]
        else:
            # no point
            pow = 0
        return reduce(lambda x, y: x * 10 + y, map(int, s)) / (10 ** pow)

    print("str2float('123.456') =", str2float("123.456"))
    if abs(str2float("123.456") - 123.456) < 0.00001:
        print("测试成功!")
    else:
        print("测试失败!")
