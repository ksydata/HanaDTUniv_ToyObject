## 상장기업의 재무제표 계정과목 데이터(연결 재무상태표, 연결 포괄손익계산서) ##
    # SY's Q. 현금흐름표 내 재무데이터도 불러올 수 있을지 의문
    # SY's Q. 연간 사업보고서, 반기보고서, 분기보고서 중 시계열을 어떻게 둘지 결정
    # Reference. CAPM(자본자산가격결정모형), Fama French Muti Factor Model(다요인모형)은 분기별 시계열
    # https://www.dinolabs.ai/388

# import dart_fss as dart
import pandas as pd
import numpy as np
import math

import re
import urllib
import requests
from bs4 import BeautifulSoup

import io
import os
import glob


# https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016
# [DART Open API Key] 148723a5dc441466805520cde2aef20fd64cc5e7
    # rcp_no, dcm_no를 자동으로 호출하는 API key( &auto=KEY )
    # 상장주식종목번호를 어떻게 불러올 것 인지 : FnGuide, pykrx module, NaverFinance

dart_url = "https://dart.fss.or.kr/pdf/download/main.do?rcp_no=20230316001364&dcm_no=9067107"
    # 하나금융지주의 rcp_number와 dcm_number "https://dart.fss.or.kr/pdf/download/main.do?rcp_no=20230316001364&dcm_no=9067107

# 1. user_agent
    # url, period, rcp_no, dcm_no
# 2. url_response = requests.get(dart_url, headers = {"user-agent":user_agent})
# 3. table = BytesIO(url_response.content)
# 4. pd.read_excel(table, skiprows = 5
# 5. data data.to_csv(str(period) + corporation + sheet + ".csv")


class FinancialDataCrawlingwithDART():

    def __init__(self):
        self.__getStockCode = getStockCode

    def dartAPIRequest(corp_code):
        # "https://opendart.fss.or.kr/api/fnlttSinglAcnt.xml?crtfc_key=148723a5dc441466805520cde2aef20fd64cc5e7&corp_code={}&bsns_year=2022& reprt_code={}"
        # 00126380 | 11011

        api_url = "https://opendart.fss.or.kr/api/company.xml?"
        crtfc_key = "148723a5dc441466805520cde2aef20fd64cc5e7"
        corp_code = self.__getStockCode()
        response = requests.get(f"{api_url}crtfc=key={crtfc_key}&corp_code={corp_code}")
        self.soup_object = BeautifulSoup(response, "lxml")

        return self.soup_object


    def getStockCode(self, stock_list: List):
        # 주식종목코드 데이터프레임 stock_code 생성
        stock_code = pd.read_html(self.html, header=0)[0]
        stock_code = stock_code[["종목코드", "회사명"]]
        stock_code["종목코드"] = stock_code["종목코드"].apply(
            lambda x: "0" * (8 - len(str(x))) + str(x))
        # 주식종목코드는 int 타입(연속형)으로 8자리에 맞추기 위한 0 추가하는 익명함수
        # [self.html] "https://kind.krx.co.kr/corpgeneral/corpList.do"

        for stocks in stock_list:
            globals()["index_{}".format(stocks)] = stock_code[stock_code["회사명"] == stocks].index[0]
            # 회사명이 self.stock인 주식종목코드가 있는 행을 index_stock에 저장함

            globals()["code_{}".format(stocks)] = stock_code.iloc[globals()["index_{}".format(stocks)], 0]
            # 입력받은 self.stock의 주식종목코드를 code_stock에 저장함
            print(stocks, globals()["code_{}".format(stocks)])

        self.stock_list = stock_list


    def downloadFinData(dart_url,
                        # rcp_no, # dcm_no,
                        period, corporation):
            # period = 20221231(or 20211231)
            # corporation = 종목코드를 html tag 내에서 웹크롤링하는 방식으로 불러오는 방식
            # 단, beautifulsoup의 object까지는 필요하지 않은 상황

        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
            # F12(ctrl+shift+l) > console > input "navigator.userAgent" > return "user-agent"

        for rcp_no, dcm_no in _:
            dart_url = "https://dart.fss.or.kr/pdf/download/main.do?rcp_no=20230316001364&dcm_no=9067107"
            url_response = requests.get(dart_url,
                                        headers = {"user-agent" : user_agent})
            table = BytesIO(url_response.content)

            for financial_sheet in ["연결 재무상태표", "연결 포괄손익계산서"]:
                data = pd.read_excel(table, financial_sheet, skiprows = 5)
                data.to_csv(str(period) + corporation + sheet + ".csv",
                            ignore_index = True,
                            encoding = "euc-kr")
                    # encoding = "cp949" or "utf-8" or "euc-kr"


