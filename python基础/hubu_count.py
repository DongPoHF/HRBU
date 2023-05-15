# str1 = input('输入英语句子:')
# str = str1.strip()
# word_count = {}
# for i in str:
#     if i in ",.!?":
#         str = str.replace(i, ' ')
# str = str.split()
# for i in str:
#     if i in word_count:
#         word_count += 1
#     else:
#         word_count = 1
# list1 = list(word_count.items())
# items = [[x, y] for (x, y) in list1]
# items.sort()
# for i in range(len(items) - 1, len(items) - 11, -1):
#     print(items[i][1] + '\t' + items[i][0])
dic = {}
s = '''Life is too short to spend time with people who suck the happiness out of you. If someone wants you in their life, they’ll make room for you. You shouldn’t have to fight for a spot. Never, ever insist yourself to someone who continuously overlooks your worth. And remember, it’s not the people that stand by your side when you’re at your best, but the ones who stand beside you when you’re at your worst that are your true friends.'''
for i in s:
    for ch in "!.,:*?":
        s = s.replace(ch, " ")
s = s.lower()
ls = s.split()
for i in ls:
    if i in dic:
        dic[i] += 1
    else:
        dic[i] = 1

print(dic)
li = list(dic.items())

# print(li[0])
# print(li[1])
li.sort(key=lambda x: x[0])
li.sort(key=lambda x: x[1], reverse=True)
print(li)
