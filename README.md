# HanaDT_SubProject
https://eunhye-zz.tistory.com/20

$(m^p)*a_1 + (m^{p-1})*a_2 + \cdot $ 계수가 1에 가까울수록 시계열 데이터는 평균에서 벗어나는 경향을 보이며, 
데이터의 계수 $m$이 1에 가까운지(있는지 없는지)를 체크하는 통계량 Augmented Dickey Fuller Test Statistics
$m=1$일 경우 단위근(unit too)이 있다고 표현하며, 단위근을 제거하기 위해 차분(differencing)하면 WN이 정상 데이터로 변환되는 수학 형태


import os
import glob
import pandas as pd
import re
from typing import *
# import tpqm


class Resource_HanaCapital():

    def __init__(self, file_path: str):
        self.file_path = file_path


    def loadExcelSheet(self, file_name: str, resource_name: str):

        resource_data = pd.DataFrame()

        for sheet_number in range(1, 29, 1):
            file_data = pd.read_excel(
                os.path.join(self.file_path + "/" + file_name),
                header = None, sheet_name = sheet_number)
                # encoding = "cp949"
            resource_data = pd.concat(
                [resource_data, file_data], axis = 0)

        resource_data.to_csv(
            os.path.join(self.file_path + "/" + resource_name),
            index = False,
            encoding = "utf-8")
            # os.path.join(self.file_path + "/" + file_name)

resource_1cap = Resource_HanaCapital(
    file_path = "C:/ResourceHanaCapital"
)
resource_1cap.loadExcelSheet(
    file_name = "사용자_권한리스트.xlsx",
    resource_name = "사용자_권한리스트_취합.csv"
)
