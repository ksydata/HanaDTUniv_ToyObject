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
    def __init__(self, file_path: str, resource_name: str, main_name: str,
                 resource_data = None, main_data = None):

        self.file_path = file_path
        self.resource_name = resource_name
        self.resource_data = pd.read_csv(
            os.path.join(self.file_path + "/" + self.resource_name),
            header = 0,
            sep = ",",
            index_col = False,
            na_values = "NA",
            encoding = "utf-8")
        self.main_name = main_name
        self.main_data = main_data


    def loadExcelSheet(self, file_name: str, resource_name: str):
        resource_data = pd.DataFrame()

        for sheet_number in range(0, 28, 1):
            # ValueError: Worksheet index 28 is invalid, 28 worksheets found
            file_data = pd.read_excel(
                os.path.join(self.file_path + "/" + file_name),
                header = None, sheet_name = sheet_number)
                # KeyError : encoding = "cp949"
            resource_data = pd.concat(
                [ resource_data, file_data.iloc[1:, :] ], axis = 0)

        resource_data.columns = [
            "USER_ID", "USER_NAME",
            "DEPT_CD", "DEPT_NM",
            "ROLE_CODE", "ROLL_NAME",
            "RESC_CODE_ORG", "RESC_NAME", "ROLL_FLAG"]
        resource_data.to_csv(
            os.path.join(self.file_path + "/" + resource_name),
            index = False,
            encoding = "utf-8")
            # os.path.join(self.file_path + "/" + file_name)


    def preprocessingData(self):
    # 하나인 화면에 대한 직무권한명 없이 하나인 화면 접근 경로만 있는 448037개 행 제거
        self.resource_data["RESC_MAIN_PATH"] = self.resource_data["RESC_CODE_ORG"].apply(
            lambda path: True if len(str(path)) <= 5 else False)
        # self.resource_data.loc[ self.resource_data["RESC_MAIN_PATH"], :].to_csv(
            # os.path.join(self.file_path + "/" + "사용자_하나인_화면권한_경로.csv"),
            # index=False,
            # encoding="utf-8")
    # 하나인 화면에 대한 직무권한명이 마지막 경로에 있는 1346018개 행 활용
        self.resource_data = self.resource_data.loc[ ~self.resource_data["RESC_MAIN_PATH"], :]
        self.resource_data.drop("RESC_MAIN_PATH", axis = 1, inplace = True)


    # 하나인 화면에 대한 권한 접근 경로 및 실제 직무권한명을 분할하여 새로운 컬럼 생성
    # str의 type : <pandas.core.strings.accessor.StringMethods object at 0x0000017F79FED850>
        self.resource_data["RESC_대분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[0]
        self.resource_data["RESC_중분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[1]
        self.resource_data["RESC_소분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[2]
        self.resource_data["RESC_하나인권한"] = self.resource_data["RESC_NAME"].str.split(pat="-").str[3]

        # NA_location = self.resource_data.loc[self.resource_data.duplicated(subset = ["RESC_소분류", "RESC_하나인권한"]), :]
        # NA_index = NA_location.index

        # self.resource_data.iloc[NA_index, self.resource_data.columns.get_loc("RESC_소분류")] = 0
            # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

        self.resource_data = self.resource_data.loc[~self.resource_data["RESC_하나인권한"].isna(), :]
        self.resource_data.to_csv(
            os.path.join(self.file_path + "/" + "사용자_하나인_화면권한_분할.csv"),
            index = False,
            encoding = "utf-8")


    def pivotTableMethod(self):

        self.main_data = pd.read_csv(
            os.path.join(self.file_path + "/" + self.main_name),
            header = 0,
            sep = ",",
            index_col = False,
            na_values = "NA",
            encoding = "utf-8")

        # table5 = self.main_data.groupby(["DEPT_NM", "ROLE_NAME", "USER_NAME"])["RESC_NAME"].size()
        # table5.to_excel(os.path.join(self.file_path + "/" + "table5.xlsx"))
            # ValueError: multiple levels only valid with MultiIndex
            # "RESC_대분류", "RESC_중분류", "RESC_소분류"

        rows_index = pd.MultiIndex.from_frame(
            self.main_data.loc[:, ["DEPT_NM", "ROLE_NAME", "RESC_하나인권한"]])
            # KeyError: ('DEPT_NM', 'ROLE_NAME')
            # 계층적 색인 생성
        columns_index = ["USER_NAME"]
        table_pivot = pd.pivot_table(
            self.main_data, index = rows_index, columns = columns_index)
        table_pivot.to_excel(os.path.join(self.file_path + "/" + "table_pivot.xlsx"))
            # ValueError: operands could not be broadcast together with shapes (1794054,) (1037141,) (1794054,)


resource_1cap = Resource_HanaCapital(
    file_path = "C:/ResourceHanaCapital",
    resource_name = "사용자_권한리스트_취합.csv",
    main_name = "사용자_하나인_화면권한_분할.csv"
)
"""
resource_1cap.loadExcelSheet(
    file_name = "사용자_권한리스트.xlsx",
    resource_name = "사용자_권한리스트_병합.csv"
)
"""
# resource_1cap.preprocessingData()
resource_1cap.pivotTableMethod()
"""
# TypeError
        # if list(self.resource_data["RESC_NAME"].str.split("-").str[:].tolist()).count() == 4:
            # TypeError: list.count() takes exactly one argument (0 given)
            
        if len( list(self.resource_data["RESC_NAME"].str.split("-").str[:].tolist()) ) == 4:
            self.resource_data["RESC_대분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[0]
            self.resource_data["RESC_중분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[1]
            self.resource_data["RESC_소분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[2]
            self.resource_data["RESC_하나인권한"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[3]
            
        elif len( list(self.resource_data["RESC_NAME"].str.split("-").str[:].tolist()) ) == 3:
            self.resource_data["RESC_대분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[0]
            self.resource_data["RESC_중분류"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[1]
            self.resource_data["RESC_하나인권한"] = self.resource_data["RESC_NAME"].str.split(pat = "-").str[2]
        
        else: pass

# ValueError
        if self.resource_data["RESC_NAME"].str.split(pat = "-").str[3] == "":
            self.resource_data["RESC_중분류"] = self.resource_data["RESC_NAME"].str.split(pat="-").str[1]
            # pd.DataFrame(self.resource_data["RESC_NAME"].str.split(pat = "-").str[3]).isna() == False:
                # ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
            # pd.DataFrame(self.resource_data["RESC_NAME"].str.split(pat = "-").str[3]).empty():
                # TypeError: 'bool' object is not callable
        else:
            self.resource_data["RESC_중분류"] = self.resource_data["RESC_NAME"].str.split(pat="-").str[1]
            self.resource_data["RESC_소분류"] = self.resource_data["RESC_NAME"].str.split(pat="-").str[2]

# AttributeError: 'str' object has no attribute 'str'
        self.resource_data["RESC_하나인권한"] = self.resource_data["RESC_NAME"].apply(
            lambda data: data.str.split(pat = "-").str[2] if data.str.split(pat="-").str[3] == " " else data)
"""
"""
        table1 = self.main_data.groupby(["DEPT_NM", "USER_NAME"])["RESC_하나인권한"].size()
        table2 = self.main_data.groupby(["DEPT_NM", "RESC_중분류"])["USER_NAME"].size()
        table3 = self.main_data.groupby(["DEPT_NM", "USER_NAME"])["RESC_중분류"].size()
        table4 = self.main_data.groupby(["DEPT_NM", "RESC_중분류", "USER_NAME"])["RESC_하나인권한"].size()
        
        table1.to_excel(os.path.join(self.file_path + "/" + "table1.xlsx"))
        table2.to_excel(os.path.join(self.file_path + "/" + "table2.xlsx"))
        table3.to_excel(os.path.join(self.file_path + "/" + "table3.xlsx"))
        table4.to_excel(os.path.join(self.file_path + "/" + "table4.xlsx"))
"""
