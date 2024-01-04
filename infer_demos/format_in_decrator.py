"""
通过decorator,将任意长度的字符加框。
要求： 1、上下用等号。2、左右用竖线和一个空格。3、如长度超过10，拆行。
"""


def deco(func):
    def inner(*arg):
        length = len(str(arg[0]))
        length = 6
        print("=" * (length + 4))

        func(*arg)

        print("=" * (length + 4))

    return inner


@deco
def target(n):
    for i in range(len(n) // 6 + 1):
        print("| {0:{1}} |".format(n[6 * i : 6 * (i + 1)], 6))


target("abcdeasasdfasdf")
