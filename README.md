# HanaDTUniv_ToyObject

# Note taking

<aside>
💡 모든 일에는 기본이 중요하다

</aside>

### 아베 마사토, 빅데이터 시대, 올바른 인사이트를 위한 통계 101 X 데이터 분석

추론통계, 가설검정, 상관과 인과, 통계 모형화, 베이즈 통계

#### **데이터 분석의 3가지 목적**
① 데이터를 요약하는 것(기술통계) ② 대상을 설명하는 것 ③ 새로 얻을 데이터를 예측하는 것
① 탐색적 자료분석 ② 확증적 자료분석
상관관계가 있다면 미지의 데이터에 대한 예측이 가능해진다. 

데이터 분석의 기반이 되는 통계학에는 확률론이 있다. 
데이터를 어떤 확률분포로부터 얻은 실현값이라고 생각해본다.
확률분포(가로축 : 확률변수, 세로축 : 발생가능성)와 확률밀도함수

$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} exp(-\frac{(x-\mu)^2}{2\sigma^2})$
$P(a \leq X \leq b) = \int_a^b f(x) dx$

#### **추론통계와 확률분포**
이론적인 확률분포는 분포의 형태를 정하는 숫자인 모수 즉, 파라미터를 가진다.
정규분포의 확률밀도함수는 $\mu, \sigma$라는 2개의 파라미터로 정해진다. 평균과 표준편차에 기준으로 두고 데이터를 나열하여 분포 내 위치로 평가하는 표준화 z값
검정통계량이 따르는 확률 분포로 $\chi^2$분포, $F$분포 등이 추론통계 계산에 나타난다. 

통계학은 오차의 학문으로 데이터의 퍼짐 정도(산포, disperation)가 클수록 힘을 발휘한다. 
수집한 데이터로 발생원을 추정하는 방법을, 추론통계라고 한다. 추론통계에는 ① 확률모형의 성질을 추정하는 통계적 추론과 ② 세운 가설과 얻은 데이터가 얼마나 들어맞는지 평가하는 가설검정이 있다. 

<aside>
<img src="https://www.notion.so/icons/checkmark_gray.svg" alt="https://www.notion.so/icons/checkmark_gray.svg" width="40px" /> 추론통계란 얻은 실현값으로 이 값을 발생시킨 확률분포를 추정하는 것
= 모집단의 일부(표본)를 분석하여 전체의 성질을 추정하는 것을 말한다.

모델링이란 수학적인 확률분포를 모집단 분포에 근사하여 현실세계를 모형을 통해 관측하는 것(일반화 범위는 각 분야 고유의 지식(도메인)에 따라 다르다)이다.
모형의 한계나 제약이 무엇인지 충분히 논의하여야 한다.
따라서 편향된 추출로써 만들어진 표본으로는 모집단을 올바르게 추정할 수 없다.

</aside>

*) 회귀분석에서 종속변수 y가 아닌 잔차 e ~ N (0 , σ2)라는 조건으로 대체할 수 있다. 

선형회귀 분석이란 선형모형을 설정하고 수집된 데이터를 이용하여 회귀계수를 추정하고(OLS 방법) t-검정이나 분산 분석에 의해 설명변수의 유의성을 검정한다. 그리고 얻어진 적합(fitted) 회귀모형에 의해 주어진 설명변수의 값에 대한 종속변수의 예측치를 얻는다.

표준화 잔차의 정규성, 등분산성, 독립성 요건 즉, 오차항의 가정을 심각하게 위반하면 t분포, F검정, MSE가 성립하지 않아 통계적 추론에 문제가 발생한다. 

#### **통계적 가설검정**
알고자 하는 대상이 전체일지라도, 실제로 데이터를 얻을 가능성이 없는 요소를 포함한 모집단은 적절하지 않다. 참고로 공통의 측정방법으로 얻은 같은 성질의 값을 변수라고 한다. 
구간폭에 따라 주는 인상 등 그래프를 보는 사람의 주관적 판단에 일임되기 때문에, 히스토그램만으로는 데이터를 정확히 기술하거나 대상을 이해하려는 목적(결론을 내기 위한 것)이 달성되지 않는다. 

<aside>
<img src="https://www.notion.so/icons/checkmark_gray.svg" alt="https://www.notion.so/icons/checkmark_gray.svg" width="40px" /> 
$s\sqrt{\frac{1}{n_A} + \frac{1}{n_B}} \leq (\bar{x_A} - \bar{x_B}) - (\mu_A - \mu_B) \leq + 2 s\sqrt{\frac{1}{n_A} + \frac{1}{n_B}}$

1. 귀무가설(H0) 수립 : $\mu_A - \mu_B = 0$

2. 검정통계량을 데이터를 통해 계산
t검정(t값), 분산분석(F값), 카이제곱검정($\chi^2$)과 같은 모집단이 수학적으로 다룰 수 있는 특정 분포를 따른다는 가정을 둔 모수 검정(parametric test)방법에 따라 다른 검정통계량을 데이터로 계산
실제표본의 값을 대입 : $\bar{x_A} - \bar{x_B}$

3. p값 계산
p-value 유의확률이란 영가설이 옳다고 가정할 때 관찰한 값이 극단적인 값이 나올 확률, 확률밀도함수(검정통계량의 이론적 확률분포)에서 t값의 바깥쪽 넓이(적분) 
신뢰구간 : 
$(\bar{x_A} - \bar{x_B}) -2 \cdot s\sqrt{\frac{1}{n_A} + \frac{1}{n_B}} \leq (\mu_A - \mu_B) \leq (\bar{x_A} - \bar{x_B}) + 2 \cdot s\sqrt{\frac{1}{n_A} + \frac{1}{n_B}}$ 
*) 불편(bias) 추정량
$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \cdot \Sigma{(x - \bar{x})^2}}$ ~ t분포 ($t = \frac{\bar{x_A} - \bar{x_B}}{s\cdot \sqrt{\frac{1}{n_A} + \frac{1}{n_B}}}$)

</aside>

분산분석(ANOVA, 레빈 검정, 바틀렛 검정)  
→ 다중비교검정(본페로니 교정, 튜키 검정, 던넷 검정, 윌리엄스 검정)


#### **비율비교**

이항검정, 이산확률분포에 이항검정의 방식을 적용한 카이제곱검정(적합도검정과 독립성검정) (얻은 출현도수(관측빈도)가 카이제곱분포라는 이론적인 비율에 따라 얻어진 것인지 가설검정)

$f(x; \nu) = \frac{1}{2^{\frac{\nu}{2}\cdot \gamma(\frac{\nu}{2})}} \cdot x^{\frac{\nu}{2}-1} \cdot e^{-\frac{x}{2}}$  

연속확률변수 $x$의 확률밀도 함수가 $f(x)$와 같을 때, $x$ ~ $\chi^2(\nu)$를 따른다. 
(확률변수 $x$는 자유도가 $\nu$인 카이제곱분포를 따른다는 의미) 
카이제곱분포는 감마분포의 특수한 형태다. 
감마분포에 $a = \frac{\nu}{2}, B = 2$를 대입하면 카이제곱분포가 된다.

$\gamma(a) = \lim_{n \to \infty} \frac{n! n^a}{a(a+1) \cdots (a+n)} = \int_0^\infty x^{a-1} \cdot e^{-x} dx$
$f(x) = \frac{1}{\gamma(a)} x^{a-1} \cdot e^{-x} dx$

감마분포란 정규분포로 설명할 수 없는 부분을 보완하기 위한 분포로 팩토리얼 함수(연속곱, 계승)를 복소수까지 확장하여 새롭게 만든 감마함수로부터 유도한 확률밀도함수를 말한다. 감마분포란 a번째 사건이 일어날 때까지 걸리는 시간에 대한 연속확률분포다. 감마분포의 기댓값과 분산은 $aB, aB^2$이다.
