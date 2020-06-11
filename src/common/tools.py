# -*-coding: utf-8 -*-

import sys
sys.path.append('..')
from src.libs.log import L
from urllib import request, parse
from http import cookiejar
from selenium import webdriver
# from pyquery import PyQuery as pq
from src.config import Config as C
# import requests


def get_cookie(url):
    cookie = cookiejar.CookieJar()
    handler=request.HTTPCookieProcessor(cookie)
    opener = request.build_opener(handler)
    response = opener.open(url)
    for item in cookie:
        print('Name = %s' % item.name)
        print('Value = %s' % item.value)

def test(url):
    cookie_dict = {}
    driver=webdriver.PhantomJS(executable_path=C.js_path)
    driver.get(url)
    # 获取cookie列表
#     cookie_list=driver.get_cookies()
#     for cookie in cookie_list:
#         cookie_dict[cookie['name']] = cookie['value']
    print(cookie_dict)
    print("=======================================")
    print(driver.page_source)
    # get_cookie(url)
    
#     driver.add_cookie(cookie_dict)
    
#     sleep(3)
#     driver.refresh()

#     driver.get(url)
#     print(driver.page_source)
    
    return
    import urllib
    import requests
    
    print(requests.get(url, headers={'User-Agent': 'Chrome'}))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3;Win64;x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    response = urllib.urlopen(url).read()
    print(response)
    
    
    return res

def request_driver(url):
    driver=webdriver.PhantomJS(executable_path=C.js_path)
    driver.get(url)
    res = driver.page_source
    return res

'''主要请求过程'''    
def request_url(url, text = False, param = {}):
    res = ''
    if 1:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
            'Cookie': '___rl__test__cookies=1580384615519; insert_cookie=57963235; OUTFOX_SEARCH_USER_ID_NCOO=960198061.7916372; ___rl__test__cookies=1580384608274',
            'If-Modified-Since': 'Thu, 20 Jan 2020 11:01:41 GMT',
            'If-None-Match': "39e3-59d5961c93058"
        }
        data = bytes(parse.urlencode(param), encoding='utf-8') 
        req = request.Request(url, data=data, headers=headers, method='GET')
        response = request.urlopen(req) 
        rst = response.read()  
        res = str(rst, encoding="utf8")


    return res 

if __name__ == '__main__':
    pass