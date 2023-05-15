# dict1 = {'name': '张三', 'age': 22}
# dict2 = {}
# # 字典的基本操作(增删改查)
# dicts = {'k1': 'v1', 'k2': 'v2'}  # 通过键去找值
# # 访问字典的元素
# print(dicts['k1'])
# # 添加字典元素
# dicts['k3'] = 'v3'
# print(dicts)
# # 修改字典元素
# dicts['k1'] = 'v001'
# print(dicts)
# # 删除字典元素
# del dicts['k1']
# print(dicts)
'''
v1
{'k1': 'v1', 'k2': 'v2', 'k3': 'v3'}
{'k1': 'v001', 'k2': 'v2', 'k3': 'v3'}
{'k2': 'v2', 'k3': 'v3'}
'''
# 字典的序列操作
# 成员检测(检测键)
# dicts = {'k1': 1, 'k2': 2, 'k3': 3}
# print('v1' not in dicts)
# # 序列函数(对键做操作)
#
# # len() 计算字典长度
# print(len(dicts))
# # max() 获取字典中最大的键
# print(max(dicts))
# # min() 获取字典中最小的键
# print(min(dicts))
'''
True
3
k3
k1
'''
# 字典的遍历
# dicts = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4, 'k5': 5}
# for key in dicts:
#     print('值', dicts[key])
#
# for key, value in dicts.items():
#     print('key:', key, 'value:', value)
#
# print(dicts.items())
'''
值 1
值 2
值 3
值 4
值 5
key: k1 value: 1
key: k2 value: 2
key: k3 value: 3
key: k4 value: 4
key: k5 value: 5
dict_items([('k1', 1), ('k2', 2), ('k3', 3), ('k4', 4), ('k5', 5)])
'''
# 字典的推导式
# 普通字典的推导式
# 变量 = {key:value for key value in 字典.items()}
# dicts = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4, 'k5': 5}
# res = {key + '+-+': value for key, value in dicts.items()}
# print(res)
# # 带有判断条件的字典推导式
# # 变量 = {key:value for key value in 字典.items() if 判断条件}
# info = {
#     'name': 'zs',
#     'age': 20
# }
# res = {k: v for k, v in info.items() if k == 'name'}
# print(res)
# # 多个循环的字典内涵
# # 变量={k1+k2: v1+v2 for k1,v1 in 字典1.items() for k2,v2 in 字典2.items() if 判断条件}
# dict1 = {}
'''
{'k1+-+': 1, 'k2+-+': 2, 'k3+-+': 3, 'k4+-+': 4, 'k5+-+': 5}
{'name': 'zs'}
'''

# 字典的专用函数
# 功能：清空字典
# 格式：字典.clear()
# 返回值：None
# 注意：直接改变原有字典
# info.clear()
info = {
    'name': 'zs',
    'age': 20
}
print(info.clear())

# 功能：复制字典
# 格式：字典.copy()
# 返回值：新的字典
info = {
    'name': 'zs',
    'age': 20
}
res = info.copy()
print(res)

# 功能：根据键获取指定的值
# 格式：字典.get(键[默认值])
# 返回值：值(默认值)
# 注意:如果键不存在,则使用默认值，如果没有默认值返回None

print(info.get('ages', 18))

# 功能：将字典的键值转化为类似元组的形式方便遍历
# 格式：字典.items()
# 返回值:类似于元组的格式
print(info.items())

# 功能：将字典的所有值组成一个序列
# 格式：字典.keys()
# 返回值：列表
print(info.values())

# pop()
# 功能：移除字典中指定的元素
# 格式：字典.pop(键[,默认值])
# 返回值：被移除的键对应的值
# 注意：如果键不存在，则报错，如果有默认值就返回默认值
print(info.pop('ages', 'NULL'))

# 功能：移除字典中的键值对
# 格式：字典.popitem()
# 返回值：键值对组成的元组
# 注意：弹出一个，原字典就少一个，字典为空不能弹出，并报错
# info.popitem()

# 功能：添加一个元素
# 格式：字典.setdefault(键，值)
# 返回值：NONE
# 注意：添加时键存在不进行任何操作，键不存在则添加
info.setdefault('school', 'hrbu')
print(info)


# update
# 功能：修改字典中的值
# 方式1：字典.update(k=v)
# 方式2：字典.update({k:v})
info.update(name='lisi')
print(info)
# def fun():
#     dicts = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4, 'k5': 5}
# # for key in dicts:
# #     print('值', dicts[key])
# #
# # for key, value in dicts.items():
# #     print('key:', key, 'value:', value)
# #
#     print(dicts.items())
# fun()
'''
None
{'name': 'zs', 'age': 20}
18
dict_items([('name', 'zs'), ('age', 20)])
dict_values(['zs', 20])
NULL
'''