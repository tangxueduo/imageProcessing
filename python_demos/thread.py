import time
from threading import Thread

import numpy as np


def func1():
    a = np.random.randint(10, size=(3, 4, 4))
    b = np.random.randint(10, size=(3, 4, 4))
    print("1 Working")


def func2():
    c = np.random.randint(10, size=(3, 4, 4))
    b = np.random.randint(10, size=(3, 4, 4))
    print("2 Working")


if __name__ == "__main__":
    t0 = time.time()
    Thread(target=func1).start()
    Thread(target=func2).start()
    print(time.time() - t0)
