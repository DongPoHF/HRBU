# # 创建列表
# list1 = list()  # 创建一个空列表
# list2 = list([1, 2, 3, 4])  # 整形列表
# list3 = list(['a', 'b', 'c', 'd'])  # 字符串列表
# list4 = list(range(1, 5))
# list5 = list('abcd')
# print(list1)
# print(list2)
# print(list3)
# print(list4)
# print(list5)
# '''
# []
# [1, 2, 3, 4]
# ['a', 'b', 'c', 'd']
# [1, 2, 3, 4]
# ['a', 'b', 'c', 'd']
# [1, 2, 'a', 'b']
# '''
# list1 = []
# list2 = [1, 2, 3, 4]
# list3 = ['a', 'b', 'c', 'd']
# list1 = [1, 2, 'a', 'b']
# print(list1)
import random


def suoyin():
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 取出第一个值
    print(list1[0])
    # 取出第二个值
    print(list1[1])
    # 取出第四个个值
    print(list1[3])


# suoyin()
'''
1
2
4
'''


def jieqv():
    list1 = [1, 2, 3, 4, 5, 'a', 'b', 'c']
    # 获取整个列表
    print(list1[:])
    # 获取列表开头到结束索引之前
    print(list1[:6])
    # 获取开始索引到列表末尾的数据组成的列表
    print(list1[0:])
    # 获取开始索引和结束索引之间的数据组成的列表
    print(list1[2:7])
    # 获取开始索引和结束索引之间的数据按照间隔值获取
    print(list1[0:9:2])


# jieqv()
'''
[1, 2, 3, 4, 5, 'a', 'b', 'c']
[1, 2, 3, 4, 5, 'a']
[1, 2, 3, 4, 5, 'a', 'b', 'c']
[3, 4, 5, 'a', 'b']
[1, 3, 5, 'b']
'''


def wain():
    list1 = [1, 2, 3, 4, 5, 6, 7]
    # start > end
    print(list1[4:2])
    # end > len(list1)
    print(list1[5:9])


# wain()
'''
[]
[6, 7]
'''


def caoz():
    list1 = [1, 2]
    list2 = [3, 4]
    list3 = list1 + list2
    # print(list3)
    list4 = list1 * 3
    print(list4)


# caoz()
'''
[1, 2, 3, 4]
[1, 2, 1, 2, 1, 2]
'''


def pand():
    list1 = [1, 4, 9, 5, 7, 'a', 'r', 's', 'b']
    a = 2 in list1
    print(a)


'''
list1 = [1, 4, 9, 5, 7, 'a', 'r', 's', 'b']
2 in list1
Out[4]: False
2 not in list1
Out[5]: True
a in list1
'a' in list1
Out[7]: True
'a' not in list1
Out[8]: False

'''


def fun():
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(len(list1))
    print(max(list1), min(list1))
    print(sum(list1))
    random.shuffle(list1)
    print(list1)


# fun()
'''
9
9 1
45
[2, 4, 3, 9, 7, 8, 5, 6, 1]
'''


def fun1():
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(0, len(list1)):
        print('下标：', i, '元素值：', list1[i])
    print('------------------')
    for u in list1:
        print('元素值:', u)


# fun1()
'''
下标： 0 元素值： 1
下标： 1 元素值： 2
下标： 2 元素值： 3
下标： 3 元素值： 4
下标： 4 元素值： 5
下标： 5 元素值： 6
下标： 6 元素值： 7
下标： 7 元素值： 8
下标： 8 元素值： 9
------------------
元素值: 1
元素值: 2
元素值: 3
元素值: 4
元素值: 5
元素值: 6
元素值: 7
元素值: 8
元素值: 9
'''


def fun2():
    list1 = [1, 2, 1, 4, 5, 6, 7, 8, 9]


'''
list1 = [1, 2, 1, 4, 5, 6, 7, 8, 9]
list1.append(10)
list1
Out[4]: [1, 2, 1, 4, 5, 6, 7, 8, 9, 10]
list1.count(1)
Out[5]: 2
list2=[11,12]
list1.extend(list2)
list1
Out[8]: [1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
list1.index(4)
Out[9]: 3
list1.insert(2,3)
list1
Out[11]: [1, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]
list1.pop(3)
Out[12]: 1
list1.pop()
Out[13]: 12
list1.reverse()
list1
Out[15]: [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
list1.sort()
list1
Out[17]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

'''


def fun3():
    # 列表内涵/列表的推导式
    book = ['水浒传', '西游记']
    books = ['《' + i + '》' for i in book]
    print(books)

    # 带有判断条件的列表推导式
    books = ['《' + i + '》' for i in book if i == '水浒传']  # 三部分 i：变量处理 for：遍历 if：判断
    print(books)

    book = ['水浒传', '西游记']
    writer = ['施耐庵', '吴承恩']
    # 多个例如同时循环的列表推导式   组合 4个
    book_info = [b + ':' + w for b in book for w in writer if b == '水浒传']
    print(book_info)
# fun3()
'''
['《水浒传》', '《西游记》']
['《水浒传》']
['水浒传:施耐庵', '水浒传:吴承恩']
'''