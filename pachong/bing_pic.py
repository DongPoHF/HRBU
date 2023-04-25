import time
import requests
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from lxml import etree
from selenium.webdriver.chrome.options import Options

# public
headers = {
    'User-Agent': 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
}
s = Service(r'D:\dongpo\chromedriver_win32\chromedriver.exe')
chrome_options = Options()
chrome_options.add_argument('--headless')

head_url = 'https://cn.bing.com'

# 构建地址池
fp = open('./pic_url.txt', 'a', encoding='utf-8')


# 第一层（负责翻页和抓取有效信息）
def index():
    # 翻页请求
    for num in range(1, 700, 35):
        base_url = 'https://cn.bing.com/images/async?q=%E9%B1%BC%E9%A6%99%E8%82%89%E4%B8%9D&first=0&count=35'
        print('采集{}页面中，并写入'.format(num))
        response = requests.get(url=base_url, headers=headers)
        response.encoding = 'utf-8'
        html = response.text
        htmls = etree.HTML(html)
        big_pic_url = htmls.xpath('//ul[@class="b_dataList"]/li/a/@href')
        for i in big_pic_url:
            full_url = head_url + i
            next_pic(full_url)


# 第二层（获取信息图片的连接）
def next_pic(full_url):
    browser = webdriver.Chrome(service=s, options=chrome_options)
    browser.get(url=full_url)
    time.sleep(0.8)
    html = browser.page_source
    img = etree.HTML(html)
    img_src = img.xpath('//div[@class="imgContainer"]//img/@src')[1]
    fp.write(img_src + '\n')
    # browser.close()


# 下载图片（鱼香肉丝、水煮肉片、红烧鸡腿）
def down_pic():
    fp = open('./pic_url.txt', 'r', encoding='utf-8')
    i = 0
    for url in fp.readlines():
        url = eval(repr(url).replace('\\n', ''))
        print(url)
        i += 1
        name = 'yxrs' + str(i)
        contens = requests.get(url=url, headers=headers).content
        with open('./data/' + name + '.jpg', 'wb') as fp:
            fp.write(contens)


if __name__ == '__main__':
    index()
    # print('写入完毕！！！')
    # down_pic()
