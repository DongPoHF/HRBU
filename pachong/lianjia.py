import time

import requests
from lxml import etree

headers_1 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
}
fp = open('./heb.txt', 'a', encoding='utf-8')
for i in range(1, 101):
    print('第{}页正在采集'.format(i))
    time.sleep(2)
    # 沈阳
    # base_url = 'https://sy.lianjia.com/zufang/pg{}/'.format(i)
    # 哈尔滨
    # base_url = 'https://hrb.lianjia.com/zufang/pg{}/'.format(i)
    # 长春
    base_url = 'https://cc.lianjia.com/zufang/pg{}/'.format(i)
    response = requests.get(url=base_url, headers=headers_1)
    response.encoding = 'utf-8'
    html = response.text
    htmls = etree.HTML(html)
    # 信息抽取
    # 房源信息
    title_list = htmls.xpath('//a[@class="twoline"]/text()')
    # 价格
    price_list = htmls.xpath('//em/text()')
    # 城区
    adress_list = htmls.xpath('//p[@class="content__list--item--des"]//a[1]/text()')
    # 标志性地点
    adress_best_list = htmls.xpath('//p[@class="content__list--item--des"]//a[2]/text()')
    # 具体小区名称
    adress_loc_list = htmls.xpath('//p[@class="content__list--item--des"]//a[3]/text()')
    md = htmls.xpath('//i/text()')
    for t, p, a, ab, al, m in zip(title_list, price_list, adress_list, adress_best_list, adress_loc_list, md):
        info = t.strip() + '\t' + p + '\t' + a + '\t' + ab + '\t' + al + '\t' + m
        fp.write(info + '\n')
fp.close()
print('ok')
