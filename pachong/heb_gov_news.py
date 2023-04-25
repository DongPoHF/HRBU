import requests
from lxml import etree

headers = {
    'Accept': 'application/xml, text/xml, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Length': '171',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Cookie': 'JSESSIONID=3C82A6F2E0BC75248C018C4E9654EBD0; _gscu_906230247=68409511fb40ew11; zh_choose_1=s; _gscbrs_906230247=1; _gscs_906230247=t69170152m257on78|pv:1',
    'Host': 'www.harbin.gov.cn',
    'Origin': 'http://www.harbin.gov.cn',
    'Pragma': 'no-cache',
    # 'Referer': 'http://www.harbin.gov.cn/col/col98/index.html?uid=517&pageNum=3',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
}
headers_1 = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
}

data = {
    'col': '1',
    'appid': '1',
    'webid': '1',
    'path': '/',
    'columnid': '98',
    'sourceContentType': '1',
    'unitid': '517',
    'webname': '哈尔滨市人民政府',
    'permissiontype': '0',
}

fp = open('./news.txt', 'a', encoding='utf-8')

for start in range(1, 3916, 45):
    end = start + 44
    print('#############起始{}-->结束{}############'.format(start, end))
    base_url = 'http://www.harbin.gov.cn/module/web/jpage/dataproxy.jsp?startrecord={}&endrecord={}&perpage=15'.format(
        start, end)
    response = requests.post(url=base_url, headers=headers, data=data)
    response.encoding = 'utf-8'
    html = response.text
    # 数据抽取
    htmls = etree.HTML(html)
    title = htmls.xpath('//a/@title')
    href = htmls.xpath('//a/@href')
    times = htmls.xpath('//span/text()')
    for i, t, ti in zip(href, title, times):
        full_url = 'http://www.harbin.gov.cn' + i
        if len(full_url) >= 75:
            response = requests.get(url=i, headers=headers_1)
        else:
            response = requests.get(url=full_url, headers=headers_1)
        response.encoding = 'utf-8'
        html = response.text
        htmls = etree.HTML(html)
        news_list = htmls.xpath('//p/text()')
        news = ''.join(news_list)
        full_info = t + '\t' + ti + '\t' + full_url + '\t' + news + '\n'
        fp.write(full_info)
fp.close()
print('ok')
