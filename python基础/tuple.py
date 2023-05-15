# t1 = ()
# t2 = (1, 2, 3)
# t3 = tuple([2 * i for i in range(1, 4)])
# t4 = tuple('abcdef')
# print(t1)
# print(t2)
# print(t3)
# print(t4)
'''
()
(1, 2, 3)
(2, 4, 6)
('a', 'b', 'c', 'd', 'e', 'f')
(1, 2, 3, 4)
(1, 2, 1, 2, 1, 2)
'''
# t1 = (1, 2)
# t2 = (3, 4)
# t3 = t1 + t2
# # print(t3)
# t4 = t1 * 3
# print(t4)
from random import random

# t1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# print(len(t1))
# print(max(t1), min(t1))
# print(sum(t1))

# t1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# print(len(t1))
# print(max(t1), min(t1))
# print(sum(t1))
# t1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# for i in range(0, len(t1)):
#     print('下标：', i, '元素值：', t1[i])
# print('------------------')
# for u in t1:
#     print('元素值:', u)
t1 = (x for x in range(0, 9, 2))
print(t1)
print(tuple(t1))
'''
<generator object <genexpr> at 0x00000229B6523C80>
(0, 2, 4, 6, 8)
'''