# HanaDT_SubProject

## 상장기업의 재무제표 계정과목 데이터(연결 재무상태표, 연결 포괄손익계산서) ##
    # SY's Q. 현금흐름표 내 재무데이터도 불러올 수 있을지 의문
    # SY's Q. 연간 사업보고서, 반기보고서, 분기보고서 중 시계열을 어떻게 둘지 결정
    # Reference. CAPM(자본자산가격결정모형), Fama French Muti Factor Model(다요인모형)은 분기별 시계열

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

def dartAPIRequest(corp_code):
    "https://opendart.fss.or.kr/api/fnlttSinglAcnt.xml?crtfc_key=148723a5dc441466805520cde2aef20fd64cc5e7&corp_code={}&bsns_year=2022& reprt_code={}"
        # 00126380 | 11011

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
                
                
# [파이썬 최종 소스코드]
# 공직윤리시스템 금융회신요청데이터 텍스트 파일 병합
# ctrl + shift + F10 키 누르면 코드 실행 (shift + F10 키는 main함수 아래의 모든 python 파일 코드 실행)
# shift + F9 키 누르면 디버깅 실행 : 코드 에러를 일으키는 버그를 찾는 작업

import os
import glob
    # 필요한 모듈 및 라이브러리 로드

# pip install pandas
    # if Error
    # pip install pandas --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org

# import pandas as pd
# from Typing import *
    # [ModuleNotFoundError] No module named 'Typing'
    # debuging 목적(dynamic -> static으로 활용방식을 변환하기 위한 객체 타입 선언)


class MergeTextFile():

    # 생성자 __init(itiate)__로 클래스 내부에 캡슐화하는 멤버변수 초기화
    def __init__(self, file_path, merge_file_name):

        # 텍스트 파일 로컬 경로
        self.file_path = file_path
            # 로컬디스크(C:) \ 디렉토리 \ 파일
            # 예시 : "C:/FTC_downloads/20230420_요청명단(심사)"
        # 새롭게 생성하는 텍스트 병합 파일 경로 및 이름
        self.merge_file_name = merge_file_name
            # "C:\FTC_downloads\20230420_요청명단(심사)\정부공직자윤리위_회신데이터.txt"

    def mergeTxT(self):
        # 텍스트 파일 n개가 포함된 C(로컬디스크) 내 파일경로 설정
        path = self.file_path
        # 파일경로 내 텍스트 파일명을 디렉토리 리스트에 저장
        directory = os.listdir(path)

        # 새롭게 생성한 텍스트 병합 파일을 저장할 경로 및 파일명을 설정
        outfile_name = self.merge_file_name
        # 아직 비어있는 새롭게 생성한 텍스트 병합 파일 내 데이터 입력 모드로 열기
        outfile = open(outfile_name, "w")


        for file_name in directory:
            # txt 텍스트 확장자의 파일이 아니라면 파일 내 데이터를 가져오지 말고 패스하여
            # 반복문 내 다음 작업을 수행
            if ".txt" not in file_name:
                continue

            # 디렉토리 내 여러 텍스트 파일 중 하나의 파일을 불러와 데이터 읽기 모드
            file = open(os.path.join(self.file_path + "/" + file_name))
            content = file.read()

            # 읽은 데이터를 텍스트 병합 파일 내 쓰기 모드
            # 단, 줄간격 1번 띄울 것
            outfile.write(content + "\n")
            # 반복문에서 디렉토리 내 여러 텍스트 파일 열기 모드를 종료
            file.close()

        outfile.close()

        # 병합된 텍스트 파일 불러오기
        merge_file = open(self.merge_file_name, "r")
            # [pandas.DataFrame] merge_file = pd.read_csv(self.merge_file_name, index_col = False)


        return merge_file



# 클래스 MergeTextFile()의 객체를 merge_instance로 선언
# 회신자료 병합본을 생성하여 통보대상을 하나인에서 매번 조회할 때
# 새롭게 입력해야하는 값은 file_path와 merge_file_name

merge_instance = MergeTextFile(
    # 텍스트 파일 로컬 경로
    file_path = "C:/FTC_downloads/20230424_요청명단(심사)",
        # 로컬디스크(C:) / 디렉토리 / 파일
        # 예시 : "C:/FTC_downloads/20230420_요청명단(심사)"
    # 새롭게 생성하는 텍스트 병합 파일 경로 및 이름
    merge_file_name = "C:/FTC_downloads/20230424_요청명단(심사)/정부공직자윤리위_회신데이터.txt"
        # 예시 : "C:/FTC_downloads/20230420_요청명단(심사)/정부공직자윤리위_회신데이터.txt"
)
merge_instance.mergeTxT()


