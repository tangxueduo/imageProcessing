import time
from functools import wraps


def timer(func):
    """
    通过decorator 显示函数运行的性能
    例如，可快速对比map和list comprehension性能差。
    要求： 除了显示CPU用时信息外，尽可能详细地显示所有性能相关信息。可以google
    """

    @wraps(func)  # <- 用于保留原函数信息
    def wraper(*args, **kwargs):
        before = time()
        result = func(*args, **kwargs)
        after = time()
        print("elapsed: ", after - before)
        return result

    return wraper
