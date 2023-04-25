# code=utf-8
import time
import re
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from lxml import etree
import random

user_agent = [
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
    "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
    "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
    "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
    "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
    "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
    "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
    "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
    "UCWEB7.0.2.37/28/999",
    "NOKIA5700/ UCWEB7.0.2.37/28/999",
    "Openwave/ UCWEB7.0.2.37/28/999",
    "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
    "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25",
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
    'Referer': 'http://www.htqyy.com'
}


def spiders(url, types, page_num):
    s = Service(r'D:\dongpo\chromedriver_win32\chromedriver.exe')
    browser = webdriver.Chrome(service=s)
    # browser.maximize_window()
    browser.get(url=url)
    time.sleep(2)
    print('#######解析页面中#######')
    html = browser.page_source
    nextPage(html, types)
    for i in range(0, page_num - 1):
        print('#######解析页面中#######')
        browser.find_element(By.CSS_SELECTOR, '.next').click()
        time.sleep(2)
        html = browser.page_source
        nextPage(html, types)
    browser.close()


# 详情信息
def nextPage(html, types):
    htmls = etree.HTML(html)
    href = htmls.xpath('//span[@class="title"]/a/@href')
    for i in href:
        full_url = 'http://www.htqyy.com/' + i
        response = requests.get(url=full_url, headers=headers)
        response.encoding = 'utf-8'
        html = etree.HTML(response.text)
        scripts = html.xpath('//body/script[3]/text()')[0]
        music_urls = scripts.split('\r\n')
        name = re.findall('var bdText = "(.*)"', music_urls[-4])[0]
        music_url = re.findall('var mp3="(.*)"', music_urls[-6])[0]
        music_types = music_url.split('/')[1]
        music_download_url = 'http://f3.htqyy.com/play9/' + music_url
        download(name, music_download_url, music_types, types)


def download(name, music_download_url, music_types, types):
    response = requests.get(url=music_download_url, headers=headers)
    headers[
        'Cookie'] = 'FPTOKEN=30$7qS3mUpGHf8VsQPReho2iVa2EvQiZYog5yFPBP17EPKBuA+8f3D5X8voEyOCgwyJoCPckbX6ClWd6HEePlsur/m5FHz64dAawdXQuEBoYM04RpaZya+bqh+u25ePDbrHtQEd8cGdbIiLAQKYzHdO1V3Acw4Ssg0z9JAcsBNTmGOTmsY3pbpQHmh8mcZUxwcNBPRRsumh4cWXOwbzSiiXdc2QoQrygbj5baZ90O3W6yxqaCoBbO/V7ROOXT+JGGoxYk3aLPFgsNLuFcNieLac/XC4DqdssSZczcLXrosDKDk8Sjji4z0kI/HbVygTYkz9pFpn72O778R6DBCx9YfAX8rsLa9DEidLg3qMEAPYYn60FqSBJ/go8t/3iR/DYjK0|3viBHdleVXFIwpswGWd2zVgIuq+XBa1AWLxyyMPnPH0=|10|0578f2b26996a4e298e678f77e6f3c9a'
    print('正在下载《{}》....'.format(name))
    fp = open('../data/music/{}.{}'.format(types, name, music_types), 'wb')
    fp.write(response.content)
    fp.close()


if __name__ == '__main__':
    # 传入采集的首页地址
    base_url = 'http://www.htqyy.com/genre/8'
    # 传入采集页面的页面数
    page_num = 2
    # 传入types
    types = 't'
    spiders(url=base_url, types=types, page_num=page_num)
