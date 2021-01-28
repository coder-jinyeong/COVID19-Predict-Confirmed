import pandas as pd # 텍스트 입출력 및 데이터프레임 가공 library
import plotly.graph_objs as go # 그래프 library
import plotly.offline as py
from fbprophet import Prophet #시계열 library
from fbprophet.plot import plot_plotly, add_changepoints_to_plot

#csv 파일 로드
df_korea = pd.read_csv(r'C:\Users\JY\OneDrive\Desktop\COVID_Confirmed.csv')

df_korea = df_korea.T[1:] # transpose
df_korea = df_korea.reset_index().rename(columns = {'index' : 'date', 0 : 'confirmed'})
df_korea['date'] = pd.to_datetime(df_korea['date'])
print(df_korea)

fig = go.Figure() #plotly 사용
fig.add_trace(
    go.Scatter(
        x = df_korea.date,
        y = df_korea.confirmed,
        name = "Confirmde in Korea"
    )
)

#prophet패키지로 예측하기위해 데이터 수정 (columns 는 ds로 예측값은 y 로)
df_prophet = df_korea.rename(columns = {
    'date' : 'ds',
    'confirmed' : 'y'
})

#300개 가량의 데이터가 있으므로 daily 와 weekly 자료 분석
m = Prophet(
    changepoint_prior_scale = 0.8, #값이 커질 수록 모델을 유연하게 설정
    changepoint_range = 0.98, #데이터 앞쪽 몇퍼센트 부분 안에서 변화점을 만들지 설정
    yearly_seasonality = False,
    weekly_seasonality = True,
    daily_seasonality = True,
    seasonality_mode = 'additive'  #계절 변동의 영향 True
)
m.fit(df_prophet) #모델 학습

future = m.make_future_dataframe(periods =10) #10일 동안의 데이터 저장공간 생성
forecast = m.predict(future) # 미래 예측하기

#결과 보기 ds : 날짜, yhat : 예측값, yhat_lower : 오차를 고려한 예측 최소값, yhat_upper : 오차를 고려한 예측 최대값
print(forecast[['ds', 'yhat', 'yhat_lower','yhat_upper']])

#그래프로 결과 보기
fig = plot_plotly(m, forecast)
fig.show()
