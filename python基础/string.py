# # str1 = 'abcdefg'
# # for i in range(len(str1) - 1, -1, -1):
# #     # print('索引：', i, '字符：', str1[i])
# #     pass
# '''
# 索引： 0 字符： a
# 索引： 1 字符： b
# 索引： 2 字符： c
# 索引： 3 字符： d
# 索引： 4 字符： e
# 索引： 5 字符： f
#
# g
# f
# '''
# # str1 = 'abcdefg'
# # print(str1[-1])
# # print(str1[-2])
#
# # str1 = 'hello world'
# # print(str1[:6])
# # print(str1[2:])
# # print(str1[5:7])
# # print(str1[::-1])
# '''
# hello
# llo world
#  w
# dlrow olleh
# '''
#
# #  首字母大写
# test_1 = 'this Is test code '
# test_1 = test_1.capitalize()
# print(test_1)
# # 每一个单词首字母大写
# test_1 = 'this Is test code '
# test_1 = test_1.title()
# print(test_1)
# # 所有单词大写
# test_1 = 'this Is test code '
# test_1 = test_1.upper()
# print(test_1)
# # 小写
# test_1 = 'this Is test code '
# test_1 = test_1.lower()
# print(test_1)
# # 大小写互换
# test_1 = 'this Is test code '
# test_1 = test_1.swapcase()
# print(test_1)
# # 测量长度
# len(test_1)
# '''
# This is test code
# This Is Test Code
# THIS IS TEST CODE
# this is test code
# THIS iS TEST CODE
# '''
# # 统计某个字符串出现的次数(词频)
# str = '午夜笛 笛声残 偷偷透 透过窗'
# str1 = str.count('笛')
# print(str1)
# # 查找 当前词汇第一个字出现的位置
# str = '午夜笛 笛声残 偷偷透 透过窗'
# str1 = str.find('偷')
# print(str1)
# # 检测是否以指定的字符开头    endswith 结尾
# str = '午夜笛 笛声残 偷偷透 透过窗'
# str1 = str.startswith('午')
# print(str1)
'''
2
8
True
'''
# 是否由纯数字组成 十进制 isdigit()
# 是否由数值字符串组成 数字整型 isnumeric()
# 是否由纯数值字符串组成  isdecimal()
# 检测字符串是否是标题模式  istitle()

# # 检测都是大写
# str = 'abcdABCD'
# str1 = str.isupper()
# print(str1)
# # 检测字符串是否由数字，字母文字等组成   isalpha  字符
# test_3 = ' !@#$%^&*'
# str1 = test_3.isalnum()
# print(str1)
#
# # 检测字符串是否由空白字符组成
# test_4 = '\n\r'
# str1 = test_4.isspace()
# print(str1)
#
# # 字符串的切割split()   返回值是列表
# str = '1234_55\n55_98766'
# str1 = str.split('_')
# print(str1)
'''
False
False
True
['1234', '55\n55', '98766']
'''
# str1 = str.splitlines()  # 换行切割

# join() list ---> str
# list_1 = ['https://www', 'baidu', 'com']
# '.'.join(list_1)
# # 填充    0填充     左填充 ljust(指定长度)  右填充 rljust
# str2 = 'aa'
# str2.zfill(11)
# str2.center(11, '*')  # 中间位填充
#
# # 去掉两侧空格
# html = '    <span class="pl">又名:</span>&nbsp;Death And All His Friends'
# html.strip()  # lstrip() 只去左  rstrip() 去右
# # html.strip('>') #去掉左右两侧 的东西 两头！！！ 两侧相同字符
# # 去掉两侧空格
# html = '    <span class="pl">又名:</span>&nbsp;Death And All His Friends  '
# str1 = html.strip()
# print(str1)
'''
<span class="pl">又名:</span>&nbsp;Death And All His Friends
'''
# 不设置指定位置，按默认顺序
# info = '我叫{},今年{}，我来自{}'.format('张三', '22', '哈尔滨')
# print(info)
'''
我叫张三,今年22，我来自哈尔滨
我叫李四,今年哈尔滨，我来自22
'''
# # 设置指定位置
# name = '李四'
# age = 22
# where = '哈尔滨'
# info = '我叫{0},今年{2}，我来自{1}'.format(name, age, where)
# print(info)
# 通过关键字的方式传递参数
# name = '李四'
# age = 22
# where = '哈尔滨'
# info = '我叫{name},今年{age}，我来自{where}'.format(name='王五', age='23', where='哈尔滨')
# print(info)
'''
我叫王五,今年23，我来自哈尔滨
张三,20,哈尔滨学院
3.15
10001
200,000,000
'''
# 通过下标的方式传递参数
# info = ['张三', 20, '哈尔滨学院']
# info_str = '{0[0]},{0[1]},{0[2]}'.format(info)
# print(info_str)
# 精度与类型f
info = '{:.2f}'.format(3.1495926)
print(info)
# 进制类型
# b--2  d--10 o--8  x--16
info = '{:b}'.format(17)
print(info)
info = '{:,}'.format(200000000)
print(info)
