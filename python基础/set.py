# s1 = set()
# s2 = {1, 4, 7, 8}
# s3 = set((4, 5, 2, 8))  # 元组转集合
# s4 = set(x * 3 for x in range(1, 11))
# print(s1)
# print(s2)
# print(s3)
# print(s4)
'''
set()
{8, 1, 4, 7}
{8, 2, 4, 5}
{3, 6, 9, 12, 15, 18, 21, 24, 27, 30}
'''
# s1 = {1, 4, 7, 2}
# s1.add(8)
# print(s1)
# print(len(s1))
# print(max(s1))
# print(min(s1))
# print(sum(s1))
# print(1 in s1)
# s1.remove(2)
# print(s1)
'''
{1, 2, 4, 7, 8}
5
8
1
22
True
{1, 4, 7, 8}
'''
# s1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# # for i in range(0, len(s1)):
# #     print('下标：', i, '元素值：', s1[i])
# # print('------------------')
# s1 = (1, 2, 3, 4, 5, 6, 7, 8, 9)
# for u in s1:
#     print('元素值:', u)
# s1 = {1, 3, 5, 7, 9}
# s2 = {2, 4, 6, 8}
# s3 = {i * 2 for i in s1}
# print(s3)
# s3 = {i * 2 for i in s1 if i > 2}
# print(s3)
# s4 = {(i, j) for i in s1 for j in s2}
# s5 = {i + j for i in s1 for j in s2}  # 可对i,j做相关操作
# print(s4)
# print(s5)
# s6 = {i + j for i in s1 for j in s2 if i > 2 and j < 8}
# print(s6)
'''
{2, 6, 10, 14, 18}
{10, 18, 6, 14}
{(3, 4), (5, 4), (9, 2), (9, 8), (1, 6), (7, 4), (5, 6), (3, 6), (9, 4), (1, 2), (1, 8), (7, 6), (3, 2), (5, 2), (3, 8), (5, 8), (9, 6), (1, 4), (7, 2), (7, 8)}
{3, 5, 7, 9, 11, 13, 15, 17}
{5, 7, 9, 11, 13, 15}
'''
# add
# 功能：向集合中添加一个元素
# 格式：集合.add(值)
# 返回值：None
# res.add(15)
# res

# pop
# 功能：删除集合中的第一个元素 类似于生成器
# 格式：集合.pop()
# 返回值：删除的那个元素
# res.pop()

# remove
# 功能：删除集合中的某个元素
# 格式：集合.remove()
# 返回值：None
# res.remove(9)
# res

# discard
# 功能：删除集合中的某个元素
# 格式：集合.discard()
# 返回值：None
# res.discard(77)
# res

# clear
# 功能：清空集合
# 格式：集合.clear()
# 返回值：None

# copy
# 功能：复制集合
# 格式：集合.copy()
# 返回值：复制的那一份集合

'''# 集合间的常规运算
set1 = {1, 2, 3, 4, 5, 6}
set2 = {4, 5, 6, 7, 8, 9}

# difference()
# 功能：差集
# 格式：集合.difference()
# 返回值：集合
# 含义：获取存在于集合1但是不存在集合2的数据的集合

print(set1.difference(set2))

# difference_update()
# 功能：差集 更新
# 格式：集合1.difference_update(集合2)
# 返回值：无 直接将结果赋值给集合2
# 含义：获取存在于集合1但是不存在集合2的数据的集合，赋值集合1
set1.difference_update(set2)
print(set1)

set1 = {1, 2, 3, 4, 5, 6}
set2 = {4, 5, 6, 7, 8, 9}
# intersection()
# 功能：交集
# 格式：集合1.intersection(集合2)
# 返回值：集合
# 含义：获取既存在于集合1又存在集合2的数据的集合
print(set1.intersection(set2))

# intersection_update()
# 功能：交集 更新
# 格式：集合1.intersection_update(集合2)
# 返回值：无 直接将结果赋值给集合2
# 含义：获取既存在于集合1又不存在集合2的数据的集合，赋值集合1
set1.intersection_update(set2)
print(set1)
set1 = {1, 2, 3, 4, 5, 6}
set2 = {4, 5, 6, 7, 8, 9}
# union()
# 功能：并集
# 格式：集合1.union(集合2)
# 返回值：集合
# 含义：将集合1与集合2中所有的数据新建一个集合(去重)
print(set1.union(set2))

# update()
# 功能：并集
# 格式：集合1.update(集合2)
# 返回值：无 将结果赋值给集合1
# 含义：将集合1与集合2中所有的数据新建一个集合(去重)，赋值
set1.update(set2)
print(set1)'''
'''
{1, 2, 3}
{1, 2, 3}
{4, 5, 6}
{4, 5, 6}
{1, 2, 3, 4, 5, 6, 7, 8, 9}
{1, 2, 3, 4, 5, 6, 7, 8, 9}
'''
# 检测一个集合是不是另外一个集合的超集
set1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
set2 = {3, 4, 5, 6}
# issuperset()
# 格式：集合1.issuperset(集合2)
# 返回值：布尔值
# set1.issuperset(set2)
#
# # issubset()
# # 格式：集合1.issubset(集合2)
# # 返回值：布尔值
# set2.issubset(set1)
#
# # isdisjoint()
# # 检测两个集合是否不相交
# # 格式：集合1.isdisjoint(集合2)
# # 返回值：布尔值
# set1.isdisjoint(set2)
# set1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
# set2 = {3, 4, 5, 6}
# # symmetric_difference()
# # 对称差集
# # 格式：集合1.symmetric_difference(集合2)
# # 返回值：集合
# # 含义：将集合1和集合2中不相交的部分取出来组成新的集合
# print(set1.symmetric_difference(set2))
#
# # symmetric_difference_update()
# # 对称差集更新
# # 格式：集合1.symmetric_difference_update(集合2)
# # 返回值：集合
# # 含义：将集合1和集合2中不相交的部分取出来组成新的集合,赋值
# set1.symmetric_difference_update(set2)
# print(set1)
'''
{1, 2, 7, 8, 9}
{1, 2, 7, 8, 9}
'''
# s1 = frozenset()
# s2 = frozenset({1, 2, 3, 4})
# s3 = frozenset([4, 5, 6, 7])
# print(s1)
# print(s2)
# print(s3)
'''
frozenset()
frozenset({1, 2, 3, 4})
frozenset({4, 5, 6, 7})
1
2
4
5
7
8
'''
s1 = frozenset({1, 2, 4, 7, 8, 5, 5, 7})
for i in s1:
    print(i)
