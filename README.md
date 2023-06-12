# HanaDT_SubProject
https://eunhye-zz.tistory.com/20

class TimeSeriesFeatureEngineering():

  def __init__(self, 
               trading_repo_path: str, bs_repo_path: str,
               # X_train = None, X_test = None, Y_train = None, Y_test = None,
               stationary_adf = None, stationary_kpss = None):
    
    self.trading_repo_path = trading_repo_path
    self.bs_repo_path = bs_repo_path
    # self.X_train = X_train 
    # self.X_test = X_test 
    # self.Y_train = Y_train 
    # self.Y_test = Y_test

    self.stationary_adf = stationary_adf
    self.stationary_kpss = stationary_kpss


# ImportDataset()
  # GetRepoTradingInfoService 클래스를 통해 불러온 2020-01-01 ~ 2023-01-01까지 
  # 환매조건부채권(RP) 건별거래, 매입증권별잔고금액 데이터 로드하여 dataframe으로 병합
  # 클래스 내부에 캡슐화하지 않고 main()에서 return값으로 생성하여 전역공간에 저장 
  def ImportDataset(self):

    CaseForTradingRepo = pd.read_csv(
        self.trading_repo_path, index_col = False, encoding = "utf-8")
    BuyRepoSecuritiesBalance = pd.read_csv(
        self.bs_repo_path, index_col = False, encoding = "utf-8")
    # dataframe = CaseForTradingRepo.merge(right = BuyRepoSecuritiesBalance, how = "left", on = "기준일자")
    dataframe = CaseForTradingRepo.copy()
    return dataframe


# NonFeatureEngineering()
# 데이터 전처리하지 않은 데이터프레임 반환(F.E한 경우와 비교 목적)
  def NonFeatureEngineering(self, date_column, dataframe):

  # 기준일자 컬럼을 date_column변수로 입력받아 날짜형 타입 변환
    dataframe[date_column] = dataframe[date_column].astype("str")
    dataframe[date_column] = dataframe[date_column].apply(
        lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
    # [AttributeError] module 'datetime' has no attribute 'strftime'
      # dataframe[date_column] = dataframe[date_column].apply(lambda x: x.strftime("%Y-%m-%d"))
      # dataframe[date_column] = dataframe[date_column].apply(lambda x: x.datetime.strftime("%Y-%m-%d"))
    # [ValueError] time data '20200110' does not match format '%Y-%m-%d'
      # dataframe[date_column] = dataframe[date_column].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    if date_column in dataframe.columns:
      # dataframe["datetime"] = pd.to_datetime(dataframe[date_column])
      dataframe["DateTime"] = pd.to_datetime(dataframe[date_column])
  # 인덱스 타입이 64비트 정수형이면, DateTime컬럼으로 변경 
    if dataframe.index.dtype == "int64":
      dataframe.set_index("DateTime", inplace = True)
    dataframe.drop(["기준일자", "RP개시일자", "RP일련번호"], axis = 1, inplace = True)

  # 범주형 변수 더미화 및 범주형 변수 코드명(식별자) 제거
    dataframe.drop(
        ["매수단기금융업종구분코드", "RP매입적용통화코드", 
         "유가증권종목종류코드", "유가증권국제인증고유번호", "유가증권국제인증고유번호코드명",
         "상환기간분류코드", "RP잔존만기구분코드", "매도단기금융업종구분코드"], axis = 1, inplace = True)
    factor_column_list: List = dataframe.select_dtypes(include = "object").columns.tolist()
    dataframe[factor_column_list] = dataframe[factor_column_list].astype("category")
    pd.get_dummies(dataframe, columns = factor_column_list)
    dataframe.drop(factor_column_list, axis = 1, inplace = True)

    dataframe_NonFE = dataframe.copy()
    dataframe_NonFE.drop("datetime", axis = 1, inplace = True)
    return dataframe_NonFE


# FeatureEngineering()
  def FeatureEngineering(self, 
                         date_column: str, dataframe: str, 
                         Y: str, decompose_method: str,
                         grouping_feature: str):
    
  # 기준일자 컬럼을 date_column변수로 입력받아 날짜형 타입 변환
    dataframe[date_column] = dataframe[date_column].astype("str")
    dataframe[date_column] = dataframe[date_column].apply(
        lambda x: datetime.datetime.strptime(x, "%Y%m%d"))
    # pd.set_option("display.max_rows", None)
    # print(dataframe.groupby([date_column]).size())

    if date_column in dataframe.columns:
      # dataframe["datetime"] = pd.to_datetime(dataframe[date_column])
      dataframe["DateTime"] = pd.to_datetime(dataframe[date_column])
  # 인덱스 타입이 64비트 정수형이면, DateTime컬럼으로 변경 
    if dataframe.index.dtype == "int64":
      dataframe.set_index("DateTime", inplace = True)
  # 인덱스 시계열 "Day" 1일 단위로 설정
    dataframe.drop(["기준일자", "RP개시일자", "RP일련번호"], axis = 1, inplace = True)
    # print(dataframe, dataframe.info())
    dataframe = dataframe.asfreq("D", method = "ffill")
      # [ValueError] cannot reindex a non-unique index with a method or limit

  # 범주형 변수 더미화 및 범주형 변수 코드명(식별자) 제거 -> 오류의 연속
    dataframe.drop(
        ["매수단기금융업종구분코드", "RP매입적용통화코드", 
         "유가증권종목종류코드", "유가증권국제인증고유번호", "유가증권국제인증고유번호코드명",
         "상환기간분류코드", "RP잔존만기구분코드", "매도단기금융업종구분코드"], axis = 1, inplace = True)
    factor_column_list: List = dataframe.select_dtypes(include = "object").columns.tolist()
    dataframe[factor_column_list] = dataframe[factor_column_list].astype("category")
    # [print(dataframe[column_name].unique()) for column_name in factor_column_list]
    
    pd.get_dummies(data = dataframe[factor_column_list])
      # pandas.get_dummies는 train 데이터의 특성을 학습하지 않기 때문에 train 데이터에만 있고 
      # test 데이터에는 없는 카테고리를 test 데이터에서 원핫인코딩 된 칼럼으로 바꿔주지 않는다.
      # 결론은 쓰지 말아야 한다?

    # dummyTransformer = OneHotEncoder()
    # dummyTransformer.fit(dataframe[factor_column_list])

    # dataframe_factor_to_dummy = pd.DataFrame()
    # for column_name in factor_column_list:
      # dummyTransformer.fit(np.asarray(dataframe[[column_name]]).reshape(-1, 1))
        # ValueError: Expected 2D array, got 1D array instead: array=['은행(신탁)' '은행(신탁)' '은행(신탁)' ... '은행(신탁)' '국내은행' '여신 금융업']
        # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
      # dataframe_factor_to_dummy = dummyTransformer.fit_transform(dataframe[column_name])
      # dataframe_factor_to_dummy = pd.DataFrame(
          # dataframe_factor_to_dummy, columns = [column_name + col for col in dummyTransformer.categories_[0]])
    # print(dataframe_factor_to_dummy)
    # pd.concat([dataframe, dataframe_factor_to_dummy], axis = 1)

    # [ValueError] Shape of passed values is (1093, 1), indices imply (1093, 12)
      # globals()["dataframe_factor_to_dummy_{}".format(column_name)] = dummyTransformer.transform(dataframe[[column_name]])
          # '<class 'scipy.sparse._csr.csr_matrix'>' 희소행렬로 반환
      # globals()["dataframe_factor_to_dummy_{}".format(column_name)]= pd.DataFrame(
          # globals()["dataframe_factor_to_dummy_{}".format(column_name)], columns = [column_name + col for col in dummyTransformer.categories_[0]])
          # TypeError: Field elements must be 2- or 3-tuples, got ''매수단기금융업종명''

    dataframe.drop(factor_column_list, axis = 1, inplace = True)
  
  # 시계열 분해 : 추세 + 계절성 + 잔차 or 추세 * 계절성 * 잔차
    decomposition = sm.tsa.seasonal_decompose(
        dataframe[Y],
        model = decompose_method)
      # sm.tsa.seasonal_decompose(model = "")
      # [input] additive, multicative
    Y_trend = pd.DataFrame(
        decomposition.trend)
    Y_trend.fillna(method = "ffill", inplace = True)
    Y_trend.fillna(method = "bfill", inplace = True)
    Y_trend.columns = ["y_trend"]
      # f{} string formating으로 변수명 입력받을 수 있으나 생략

    Y_seasonal = pd.DataFrame(
        decomposition.seasonal)
    Y_seasonal.fillna(method = "ffill", inplace = True)
    Y_seasonal.fillna(method = "bfill", inplace = True)
    Y_seasonal.columns = ["y_seasonal"]

    pd.concat([dataframe, Y_trend, Y_seasonal], axis = 1).isnull().sum()
    if "y_trend" not in dataframe.columns:
      if "y_seasonal" not in dataframe.columns:
        dataframe = pd.concat([dataframe, Y_trend, Y_seasonal], axis = 1)

  # 특정 시점을 기준으로 전후 12일씩 총 24일간의 이동평균된 pandas Series타입
    Y_day = dataframe[[Y]].rolling(24).mean()
    Y_day.fillna(method = "ffill", inplace = True)
    Y_day.fillna(method = "bfill", inplace = True)
    Y_day.columns = ["y_day"]

    Y_week = dataframe[[Y]].rolling(24 * 7).mean()
    Y_week.fillna(method = "ffill", inplace = True)
    Y_week.fillna(method = "bfill", inplace = True)
    Y_week.columns = ["y_week"]

    if "y_day" not in dataframe.columns:
      if "y_week" not in dataframe.columns:
        dataframe = pd.concat([dataframe, Y_day, Y_week], axis = 1)

  # 정상성(stationarity)을 나타내지 않는 시계열의 정상성을 나타내도록, 추세나 계절성을 완화하는 차분
  # 시계열의 수준에서 나타나는 변화를 제거하여 시계열의 평균 변화를 일정하게 만드는데 도움이 된다.
  # 다시 말해 비정상적 시계열은 누적 과정(integrated procss)이기 때문에 발생할 수 있다. 
  # (https://otexts.com/fppkr/decomposition.html)
    Y_differencing = dataframe[[Y]].diff()
    Y_differencing.fillna(method = "ffill", inplace = True)
    Y_differencing.fillna(method = "bfill", inplace = True)
      # [관측값의 차이, 차분] y_t = y_{t-1} + e_t
      # [2차 차분] ( y_t - y_{t-1} ) - ( y_{t-1} - y_{t-2} )
      # 2차 이상의 차분을 한 데이터로 적합한 모델의 설명력이 낮아질 수 있다. 
      # [계절성 차분] y_t - y_{t-m} (단, m은 계절 수)
    Y_differencing.columns = ["y_difference"]
    if "y_difference" not in dataframe.columns:
      dataframe = pd.concat([dataframe, Y_differencing], axis = 1)

  # MMMM-YY-DD HH (Week) 분할
    # dataframe[f"{grouping_feature}group"] = pd.cut(dataframe[grouping_feature], 10)
    
    dataframe["datetime"] = dataframe.index
    dataframe["Year"] = dataframe["datetime"].dt.year
    dataframe["Quarter"] = dataframe["datetime"].dt.quarter
    dataframe["Quarter_version2"] = dataframe["Quarter"] + (dataframe["Year"] - dataframe["Year"].min()) * 4
    dataframe["Day"] = dataframe["datetime"].dt.day
    dataframe["DayofWeek"] = dataframe["datetime"].dt.dayofweek

  # 데이터 관측시점들 간의 시차(lagging) 효과 반영
    dataframe["Y_lag1"] = dataframe[Y].shift(1)
    dataframe["Y_lag2"] = dataframe[Y].shift(2)
    dataframe["Y_lag1"].fillna(method = "bfill", inplace = True)
    dataframe["Y_lag2"].fillna(method = "bfill", inplace = True)

  # 분기 더미변수 생성
    if "Quarter" not in dataframe.columns:
      if "QuarterDummy" not in ["_".join(column.split("_")[:2]) 
                                for column in dataframe.columns]:
          dataframe = pd.concat(
              [dataframe, pd.get_dummies(dataframe["Quarter"], prefix = "QuarterDummy", drop_first=True)], 
              axis=1)
          del dataframe["Quarter"]

    dataframe_FE = dataframe.copy()
    dataframe_FE.drop("datetime", axis = 1, inplace = True)
    return dataframe_FE


# DataSplitofCrossSectional()
# 시계열을 고려하지 않고 분할한 훈련용, 검증용 종속변수(타겟), 독립변수(피쳐) 데이터 반환 
  def DataSplitofCrossSectional(self, 
                                cleandataframe: pd.DataFrame, Y_colname: str, X_colname: List, 
                                test_ratio: float, seed):
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        cleandataframe[X_colname], cleandataframe[Y_colname], test_size = test_ratio, random_seed = seed)
    
    print(f"X_train : {X_train.shape}", f"Y_train : {Y_train.shape}")
    print(f"X_test : {X_test.shape}", f"Y_test : {Y_test.shape}")

    return X_train, X_test, Y_train, Y_test


# DataSplitofTimeSeries()
# 시계열을 고려하여 분할한 훈련용, 검증용 종속변수(타겟), 독립변수(피쳐) 데이터 반환 
  def DataSplitofTimeSeries(self, 
                            cleandataframe: pd.DataFrame, Y_colname: str, X_colname: List, 
                            criteria):
    
    dataframe_train = cleandataframe.loc[cleandataframe.index < criteria, :]
    dataframe_test = cleandataframe.loc[cleandataframe.index >= criteria, :]

    Y_train = dataframe_train[Y_colname]
    X_train = dataframe_train[X_colname]
    Y_test = dataframe_test[Y_colname]
    X_test = dataframe_test[X_colname]

    print(f"X_train : {X_train.shape}", f"Y_train : {Y_train.shape}")
    print(f"X_test : {X_test.shape}", f"Y_test : {Y_test.shape}")
    
  # 캡슐화시키지 않는 이유는 입력값을 F.E 유무에 따라 각각 투입하여 나온 결과를 비교하기 위함
    # self.X_train = X_train, 
    # self.X_test = X_test, 
    # self.Y_train = Y_train, 
    # self.Y_test = Y_test

    # return self
    return X_train, X_test, Y_train, Y_test


# Evaluationof1PairofSet()
  def Evaluationof1PairofSet(self, Y_refer, Y_pred, graph_on = False):

    loss_length = len(Y_refer.values.flatten()) - len(Y_pred)
    if loss_length != 0:
        Y_refer = Y_refer[loss_length:]

    if graph_on == True:
        pd.concat(
            [Y_refer, pd.DataFrame(Y_pred, index = Y_refer.index, columns = ["prediction"])], axis=1
        ).plot(kind = "line", figsize = (20, 6), 
               xlim = (Y_refer.index.min(), Y_refer.index.max()),
               linewidth = 3, fontsize = 20)
      
        plt.title("Time Series of Target", fontsize = 20)
        plt.xlabel("Index", fontsize = 15)
        plt.ylabel("Target Value", fontsize = 15)

  # 오차의 절댓값 평균
    MAE = abs(Y_refer.values.flatten() - Y_pred).mean()
  # 오차의 표준편차 평균
    MSE = ((Y_refer.values.flatten() - Y_pred)**2).mean()
  # 오차의 절댓값 백분율 평균
    MAPE = (abs(Y_refer.values.flatten() - Y_pred) / Y_refer.values.flatten() * 100).mean()
      
    Score = pd.DataFrame([MAE, MSE, MAPE], index=["MAE", "MSE", "MAPE"], columns = ["Score"]).T
    Residual = pd.DataFrame(Y_refer.values.flatten() - Y_pred, index = Y_refer.index, columns = ["Error"])
      
    return Score, Residual


  def EvaluationofTrainTestPairs(self, 
                                 Y_refer_train, Y_pred_train, Y_refer_test, Y_pred_test,
                                 graph_on):

  # 클래스 내부 멤버함수를 캡슐화하는 방법
    Score_train, Residual_train = self.Evaluationof1PairofSet(
        Y_refer_train, Y_pred_train, graph_on = graph_on)
    Score_test, Residual_test = self.Evaluationof1PairofSet(
        Y_refer_test, Y_pred_test, graph_on = graph_on)
    Score_train_test = pd.concat([Score_train, Score_test], axis=0)
    Score_train_test.index = ["Train", "Test"]
    
    # return self
    return Score_train_test, Residual_train, Residual_test


## 리뷰할 것(이게 최선일까) ##

# StationaryADFTest()
# 시계열의 정상성(Stationarity) 검정 : 관측된 시간과 시계열의 특징은 무관할까
# Ha : 시계열에 단위근이 존재하지 않는다. (시계열이 정상성을 만족한다)
# https://www.statsmodels.org/dev/_modules/statsmodels/tsa/stattools.html#adfuller
  def StationaryADFTest(self, 
                        Y_data # , target_name
                        ):

  # target_name(y)가 지정되어 있지 않을 경우
  # if len(target_name) == 0:

      # stationary_adf = pd.Series(sm.tsa.stattools.adfuller(Y_data)[0:4], index = ["Test Statistics", "P-value", "Used Lag", "Number of Observations Used"])

      # for key, value in sm.tsa.stattools.adfuller(Y_data)[4].items():
        # stationary_adf["Critical Values (%s)" %key] = value
          # 임계값(임계값 이상의 확률(극단적 방향으로 끝까지의 확률)이 a(유의수준)가 되는 자리값)
        # stationary_adf["Maximum Information Criteria"] = sm.tsa.stattools.adfuller(Y_data)[5]
          # Critical Value (5%)
        # stationary_adf = pd.DataFrame(
            # stationary_adf, columns = ["stationarity_adf"])

  # target_name(y)가 명시되어 있을 경우
    # else:

    stationary_adf = pd.Series(
        sm.tsa.stattools.adfuller(Y_data.values)[0:4],
        index = ["Test Statistics", "P-value", "Used Lag", "Used Observations"])
          # y_t = a_1*y_t-1 + a_2*y_t-2 + ... + e_t
          # [단위근(unit root)] t시점의 확률변수는 t-1, t-2, ... 시점의 확률변수와 관계가 있으며 에러가 포함되는 것
          # m**p - m**(p-1)*a_1 - m**(p-2)*a_2 - ... - a_p = 0
          # [m = 1] 위 식의 근이 되는 m = 1이면 시계열 확률 과정은 단위근을 가진다고 말한다. 
          # 그렇지 않다면, 약 정상성을 띠는 t시점 시계열 데이터는 확률 과정의 성질(E(Xt), Var(Xt))이 변하지 않는다.

    for key, value in sm.tsa.stattools.adfuller(Y_data.values)[4].items():
      stationary_adf["Critical Values(%s)"%key] = value
      stationary_adf["Maximum Information Criteria"] = sm.tsa.stattools.adfuller(Y_data.values)[5]
      stationary_adf = pd.DataFrame(
          stationary_adf, columns = ["stationarity_adf"])
        
    self.stationary_adf = stationary_adf
    # return self
    
  
# StationaryKPSSTest()
# Ha : 시계열에 단위근이 존재한다. (시계열이 정상성을 만족하지 않는다)
# 분산이 변하거나 계절성이 있는 시계열에 대한 정상성을 제대로 검정하지 못하는 ADF 검정과 달리
# KPSS 검정은 추세가 있거나, 분산이 변하거나, 계절성이 있는 시계열에 대하여 정상성 여부를 검정할 수 있다.
  def StationaryKPSSTest(self,
                         Y_data # , target_name
                         ):

    # if len(target_name) == 0:
      # stationary_kpss = pd.Series(sm.tsa.stattools.kpss(Y_data)[0:3], index = ["Test Statistics", "p-value", "Used Lag"])
      # for key, value in sm.tsa.stattools.kpss(Y_data)[3].item():
        # stationary_kpss["Critical Value (%s)" %key] = value
        # stationary_kpss = pd.DataFrame(stationary_kpss, columns = ["stationary_kpss"])
    
    # else:
    stationary_kpss = pd.Series(
        sm.tsa.stattools.kpss(Y_data.values)[0:3], 
        index = ["Test Statistics", "p-value", "Used Lag"])
    for key, value in sm.tsa.stattools.kpss(Y_data.values)[3].item():
      stationary_kpss["Critical Value (%s)" %key] = value
      stationary_kpss = pd.DataFrame(
          stationary_kpss, columns = ["stationary_kpss"])
      stationary_kpss = self.stationary_kpss
          
    # return self


## 리뷰할 것(이게 최선일까) ##

# ErrorAnalysis()
  def PlotErrorAnalysis(self, Y_data, X_data, graph_on = False):

    # for x in target_name:
      # target_name = x
    # X_data = X_data.loc[Y_data.index]
    if graph_on == True:
      Y_data["RowNum"] = Y_data.reset_index().index

    # Stationarity(Trend) Analysis : Plotting
      sns.set(
          palette = "muted", color_codes = True, font_scale = 2)
      sns.lmplot(
          x = "RowNum", y = "Error", data = Y_data, 
            # ["RowNum"] Y_data.reset_index().index
            # ["RP이율"] Y_data.name
          fit_reg = "True", aspect = 2, ci = 99, sharey = True)
            # [fit_reg] regression fit model relationg X and y
            # [aspect] aspect ratio of each facet (aspect * height) = width
            # [sharey] facets will share y axes across each columns(X axes across rows)
      del Y_data["RowNum"]
      # Y_data = pd.Series(Y_data)
        # ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().

    # Normal Distribution Analysis : Plotting
      figure, axes = plt.subplots(figsize = (12, 8))
      sns.distplot(Y_data.values, norm_hist = "True", fit = stats.norm, ax = axes)

    # Lag Analysis : Plotting
      length = int(len(Y_data["Error"] / 10))
      figure, axes = plt.subplots(1, 4, figsize = (12, 3))
      pd.plotting.lag_plot(Y_data["Error"], lag = 1, ax = axes[0])
      pd.plotting.lag_plot(Y_data["Error"], lag = 5, ax = axes[1])
      pd.plotting.lag_plot(Y_data["Error"], lag = 10, ax = axes[2])
      pd.plotting.lag_plot(Y_data["Error"], lag = 50, ax = axes[3])

    # Autocorrelation Analysis : Plotting
      figure, axes = plt.subplots(2, 1, figsize = (12, 5))
      sm.tsa.graphics.plot_acf(
          Y_data["Error"], lags = 100, use_vlines = True, ax = axes[0])
      sm.tsa.graphics.plot_pacf(
          Y_data["Error"], lags = 100, use_vlines = True, ax = axes[1])

    return self


  def TimeSeriesConditinalTest(self, 
                               Y_data, X_data):

  # 정상성 검정(Null Hypothesis: The Time-series is non-stationalry)
    # Stationarity_adf = self.StationaryADFTest(Y_data)
    # Stationarity_kpss = self.StationaryADFTest(Y_data)
      # [TypeError] cannot concatenate object of type '<class '__main__.TimeSeriesFeatureEngineering'>'; only Series and DataFrame objs are valid

  # 정규성 검정(Null Hypothesis: The residuals are normally distributed)
    Normality = pd.DataFrame([stats.shapiro(Y_data)],
                             index=['Normality'], columns=['Test Statistics', 'p-value']).T

  # 시계열(시차)의 자기상관성 검정(Null Hypothesis: Autocorrelation is absent) : lag(시차) 1일, 5일, 10일, 50일
    print(pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Y_data.values, lags = [1, 5, 10, 50])))
        #          lb_stat      lb_pvalue
        # 1     522.692241  1.098531e-115
        # 5    2219.759093   0.000000e+00
        # 10   3922.340298   0.000000e+00
        # 50  12305.292684   0.000000e+00
    # Autocorrelation = pd.concat(
        # [pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Y_data.values, lags = [1, 5, 10, 50]).iloc[:,0], columns = ["Test Statistics"]),
        # pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Y_data.values, lags = [1, 5, 10, 50]).iloc[:,1], columns = ["p-value"])], 
        # axis=1
    # ).T
      # pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
      # keyerror 0
    # Autocorrelation.columns = ['Autocorr(lag1)', 'Autocorr(lag5)', 'Autocorr(lag10)', 'Autocorr(lag50)']
      # ValueError: Length mismatch: Expected axis has 0 elements, new values have 4 elements

  # 등분산성 검정(Heteroscedasticity(이분산). Null Hypothesis: Error terms are homoscedastic)
    Heteroscedasticity = pd.DataFrame([sm.stats.diagnostic.het_goldfeldquandt(Y_data, X_data.values, alternative = "two-sided")],
                                      index=['Heteroscedasticity'], columns=['Test Statistics', 'p-value', 'Alternative']).T

    Score = pd.concat([self.stationary_adf, self.stationary_kpss, 
                       Normality, # Autocorrelation, 
                       Heteroscedasticity], join = "outer", axis = 1)
      # [TypeError] cannot concatenate object of type '<class '__main__.TimeSeriesFeatureEngineering'>'; only Series and DataFrame objs are valid

    index_new = ["Test Statistics", "p-value", "Alternative", "Used Lag", "Used Observations",
                 "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)", "Maximum Information Criteria"]
    Score.reindex(index_new)
    print(Stationarity_adf, Stationarity_kpss, Score)

    return Score
