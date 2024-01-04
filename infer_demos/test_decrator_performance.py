import time
from functools import wraps


def timer(func):
    """
    通过decorator 显示函数运行的性能
    例如，可快速对比map和list comprehension性能差。
    要求： 除了显示CPU用时信息外，尽可能详细地显示所有性能相关信息。可以google
    """

    @wraps(func)  # <- 用于保留原函数信息
    def inner(*args, **kwargs):
        before = time.time()
        cpu_start = time.perf_counter()
        result = func(*args, **kwargs)
        after = time.time()
        cpu_end = time.perf_counter()
        print("elapsed: ", after - before, "cpu elapsed: ", cpu_end - cpu_start)
        return result

    return inner


@timer
def use_list(n: int):
    return filter(lambda x: n % x == 0, range(1, n))


@timer
def use_list_comprehension(n: int):
    return [i for i in range(1, n) if n % i == 0]


use_list(1024)
use_list_comprehension(1024)
