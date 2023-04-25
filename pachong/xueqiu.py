# code=utf-8
import requests
import json
import re
import time

headers = {
    'cookie': 's=da13jj00ye; device_id=cf77119feac71ee94963ab0970cc9f7a; __utmz=1.1669627209.1.1.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; acw_tc=2760827916703977504122117e77e5d7598468418cfdbdb5efda9f5ea6a067; xq_a_token=df4b782b118f7f9cabab6989b39a24cb04685f95; xqat=df4b782b118f7f9cabab6989b39a24cb04685f95; xq_r_token=3ae1ada2a33de0f698daa53fb4e1b61edf335952; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOi0xLCJpc3MiOiJ1YyIsImV4cCI6MTY3MjE4Njc1MSwiY3RtIjoxNjcwMzk3NzI5MjgzLCJjaWQiOiJkOWQwbjRBWnVwIn0.kWYm2aveZZ8OcjAk-0csh6GkU-Gc2CLe30t5NHaAXDqNg_Dvq53ulqM4qjcHFqT9OOAI77pccklMowZuZYZvijH7IEX_OByZ9L3WOv6RqaVqZWdac795IM1HKMFAQO-hTCqgS7biOK9ZzBPG0-dJrSOGXt17mu7DZ0bKSnI4BZ2H28GMcQUrk28boIUbP69Sq40e70EKLAk78fBt22SP6KeaFWCEx-hOH1YwGs7Y_LNhANLD_wfYCfZjFUKXzgsoe3KXOZ1tibgKKYK7UrT65x4qk1Po5V4_8YE_V04S0tMHgdljoN6lIm19kneF8KkPTM7ex8FJK6u3UkBd_ZtNvQ; u=391670397750462; Hm_lvt_1db88642e346389874251b5a1eded6e3=1669627209,1670397753; __utma=1.1095084468.1669627209.1669627209.1670397769.2; __utmc=1; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1670398516; __utmt=1; __utmb=1.3.10.1670397769',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36',
}
fp = open('./xueqiu.txt', 'a', encoding='utf-8')
base_url = 'https://stock.xueqiu.com/v5/stock/screener/quote/list.json?page=1&size=5000&order=desc&orderby=percent&order_by=percent&market=CN&type=sh_sz'
jsons = requests.get(url=base_url, headers=headers).text
jsons_dict = json.loads(jsons)
for lists in jsons_dict['data']['list']:
    symbol = lists['symbol']  # 股票代码
    name = lists['name']  # 股票名称
    current = lists['current']  # 当前价
    chg = lists['chg']  # 涨跌额
    first_percent = lists['first_percent']
    volume = lists['volume']  # 成交量
    amount = lists['amount']  # 成交额
    turnover_rate = lists['turnover_rate']  # 换手率
    market_capital = lists['market_capital']  # 市值
    full_info = symbol + ';' + name + ';' + str(current) + ';' + str(chg) + ';' + str(first_percent) + ';' + str(
        volume) + ';' + str(amount) + ';' + str(turnover_rate) + ';' + str(market_capital) + '\n'
    fp.write(full_info)
fp.close()
print('ok')
