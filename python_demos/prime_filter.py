import math
from functools import reduce


# 100以内质数的平方根的和
def is_prime(n):
    primes = filter(
        lambda x: not [x % i for i in range(2, int(math.sqrt(x)) + 1) if x % i == 0],
        range(2, n + 1),
    )
    print(list(primes))
    return list(
        filter(
            lambda x: not [
                x % i for i in range(2, int(math.sqrt(x)) + 1) if x % i == 0
            ],
            range(2, n + 1),
        )
    )


def sum(x, y):
    return x + y


rs = round(reduce(sum, list(map(lambda x: math.sqrt(x), is_prime(100)))), 2)
# rs = round(
#     reduce(
#         lambda x, y: x + y,
#         list(
#             map(
#                 lambda x: math.sqrt(x),
#                 list(
#                     filter(
#                         lambda x: not list(
#                             filter(
#                                 lambda i: x % i == 0, range(2, int(math.sqrt(x)) + 1)
#                             )
#                         ),
#                         range(2, 100 + 1),
#                     )
#                 ),
#             )
#         ),
#     ),
#     2,
# )
print(rs)

# rs = list(filter(lambda x: not list(filter(lambda y: x%y==0,range(2, int(math.sqrt(x))))), range(2,101)))
