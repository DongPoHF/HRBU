# code=utf-8
import requests
from lxml import etree
import time
import re

# 页码分析
'''
    每隔 20 一翻页
    末页 页码为  580
'''

# public
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Cookie': 'll="118146"; bid=v8DDMSB4rb8; __gads=ID=f3fb77d625b1f9e3-2282509f65d800df:T=1668477035:RT=1668477035:S=ALNI_MZcWibtLSuL53uyPbo5B4bRgUSvqQ; __utma=30149280.114203343.1668477033.1668479060.1668578741.3; __utmc=30149280; __utmz=30149280.1668578741.3.2.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utma=223695111.1641925483.1668578752.1668578752.1668578752.1; __utmb=223695111.0.10.1668578752; __utmc=223695111; __utmz=223695111.1668578752.1.1.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1668578752%2C%22https%3A%2F%2Fwww.douban.com%2F%22%5D; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __gpi=UID=00000b7c21b80b11:T=1668477035:RT=1668578752:S=ALNI_MaGGbp2sNCBkIV8-iLMn7WVZT-FVQ; _vwo_uuid_v2=D211C5047A211B801FAE1B135F97CE491|f5d1f25e03bfc1fcca92fb85f12dfb7a; ct=y; __yadk_uid=Je7s1XXDpkeMMkj2qoOTkKQhi6ttBVJE; __utmt=1; dbcl2="264499730:A8axShho0s4"; ck=dkPT; push_noty_num=0; push_doumail_num=0; __utmv=30149280.26449; __utmb=30149280.15.10.1668578741; _pk_id.100001.4cf6=f2581d187a1e0fbd.1668578752.1.1668581395.1668578752.'
}
fp = open('./douban.txt', 'a', encoding='utf-8')

for page in range(0, 581, 20):
    base_url = 'https://movie.douban.com/subject/35160920/comments?start={}&limit=20&status=P&sort=new_score'.format(
        page)
    time.sleep(2)
    print('第{}页正在采集'.format(page))
    response = requests.get(url=base_url, headers=headers)
    response.encoding = 'utf-8'
    html = response.text
    htmls = etree.HTML(html)
    times = htmls.xpath('//span[@class="comment-time "]/@title')
    loc = htmls.xpath('//span[@class="comment-location"]/text()')
    short = htmls.xpath('//span[@class="short"]/text()')
    for t, l, s in zip(times, loc, short):
        s = repr(s).replace('\\n', '')
        full_info = t + ';' + l + ';' + s + '\n'
        fp.write(full_info)

fp.close()
print('ok')
