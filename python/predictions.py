import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 读取并预处理数据
df = pd.read_csv(
    r"D:\python\pytest\601006.csv",
    parse_dates=['Date'],
    date_parser=lambda x: pd.to_datetime(x.strip(), format='%b %d %Y'),
    index_col='Date'
)
df.index = pd.to_datetime(df.index).tz_localize(None)
series = df['Adj Close'].dropna()
# 1. 指定默认字体为黑体（SimHei），确保已安装该字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 2. ARIMA 预测（12 周）
# 假设差分与参数调优已完成，直接用最优模型
model_arima = ARIMA(series, order=(3,1,1)).fit()
pred_arima = model_arima.forecast(steps=12).values

# 3. Prophet 预测（12 周）
df_prophet = series.reset_index().rename(columns={'Date':'ds','Adj Close':'y'})
m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False)
m_prophet.fit(df_prophet)
future = m_prophet.make_future_dataframe(periods=12, freq='W')
forecast_prophet = m_prophet.predict(future)
mask = forecast_prophet['ds'] > df_prophet['ds'].max()
pred_prophet = forecast_prophet.loc[mask, 'yhat'].values[:12]

# 4. LSTM 预测（12 周）
# 4.1 构造滑动窗口数据
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.values.reshape(-1,1))
def create_seq(data, look_back):
    X,y = [],[]
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i,0]); y.append(data[i,0])
    return np.array(X), np.array(y)
look_back = 6
X, y_all = create_seq(scaled, look_back)
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_all[:split], y_all[split:]
X_train = X_train.reshape(-1,look_back,1)
X_test  = X_test.reshape(-1,look_back,1)

# 4.2 模型训练
model_lstm = Sequential([LSTM(50,input_shape=(look_back,1)), Dropout(0.2), Dense(1)])
model_lstm.compile('adam','mse')
model_lstm.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=0)

# 4.3 预测并反归一化
y_pred = model_lstm.predict(X_test).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

# 5. 提取最后12点用于对比
actual_arima   = series.iloc[-12:].values
pred_arima     = pred_arima
actual_prophet = df_prophet['y'].iloc[-12:].values
pred_prophet   = pred_prophet
actual_lstm    = y_test_actual[-12:]
pred_lstm      = y_pred_actual[-12:]

# 6. 绘制预测曲线对比
time_index = series.index[-12:]

plt.figure(figsize=(12,6))
plt.plot(time_index, actual_lstm,    label='真实收盘价', color='black')
plt.plot(time_index, pred_arima,     label='ARIMA 预测', linestyle='--')
plt.plot(time_index, pred_prophet,   label='Prophet 预测', linestyle='-.')
plt.plot(time_index, pred_lstm,      label='LSTM 预测', linestyle=':')
plt.title("大秦铁路周线收盘价：各模型预测对比")
plt.xlabel("日期")
plt.ylabel("收盘价（元）")
plt.legend()
plt.grid(True)
plt.show()


#残差分布
# 计算 LSTM 残差
residuals_lstm = y_test_actual.flatten() - y_pred_actual.flatten()

plt.figure(figsize=(8,4))
plt.hist(residuals_lstm, bins=20, edgecolor='k')
plt.title("LSTM 模型残差分布直方图")
plt.xlabel("预测误差")
plt.ylabel("频数")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ————————————————————
# 模型预测数据准备（确保长度一致为12）
actual_arima = series.iloc[-12:].values
forecast_arima_result = model_arima.get_forecast(steps=12)
forecast_arima_actual = forecast_arima_result.predicted_mean.values
ci_arima = forecast_arima_result.conf_int()
arima_lower = ci_arima.iloc[:, 0].values
arima_upper = ci_arima.iloc[:, 1].values

actual_prophet = df_prophet['y'].iloc[-12:].values
future_mask = forecast_prophet['ds'] > df_prophet['ds'].max()
prophet_future = forecast_prophet.loc[future_mask].iloc[:12]
forecast_prophet_actual = prophet_future['yhat'].values
prophet_lower = prophet_future['yhat_lower'].values
prophet_upper = prophet_future['yhat_upper'].values

actual_lstm = y_test_actual.flatten()
forecast_lstm_actual = y_pred_actual.flatten()


# ————————————————————
# 统一时间轴
time_index = df.index[-12:]

# ————————————————————
# LSTM 置信区间绘图
def mc_dropout_predict(model, X, n_iter=100):
    preds = np.array([model.predict(X, batch_size=32, verbose=0) for _ in range(n_iter)])
    mu = preds.mean(axis=0).flatten()
    sigma = preds.std(axis=0).flatten()
    return mu, sigma

mu, sigma = mc_dropout_predict(model_lstm, X_test, n_iter=200)
mu_inv = scaler.inverse_transform(mu.reshape(-1, 1)).flatten()
upper = scaler.inverse_transform((mu + 1.96 * sigma).reshape(-1, 1)).flatten()
lower = scaler.inverse_transform((mu - 1.96 * sigma).reshape(-1, 1)).flatten()

# 取最后12点
mu_inv = mu_inv[-12:]
upper = upper[-12:]
lower = lower[-12:]
true_actual = actual_lstm[-12:]

plt.figure(figsize=(12,6))
plt.plot(time_index, true_actual, label='真实收盘价', color='black')
plt.plot(time_index, mu_inv, label='LSTM 均值预测', linestyle=':')
plt.fill_between(time_index, lower, upper, alpha=0.3, label='95% 置信区间')
plt.title("LSTM 模型预测及95%置信区间")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.legend()
plt.grid(True)
plt.show()

# ————————————————————
# ARIMA 置信区间图
plt.figure(figsize=(12,6))
plt.plot(time_index, actual_arima, label='真实收盘价', color='black')
plt.plot(time_index, forecast_arima_actual, label='ARIMA 预测', linestyle='--')
plt.fill_between(time_index, arima_lower, arima_upper, alpha=0.3, label='95% 置信区间')
plt.title("ARIMA 模型预测及95%置信区间")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.legend()
plt.grid(True)
plt.show()

# ————————————————————
# Prophet 置信区间图
plt.figure(figsize=(12,6))
plt.plot(time_index, actual_prophet, label='真实收盘价', color='black')
plt.plot(time_index, forecast_prophet_actual, label='Prophet 预测', linestyle='-.')
plt.fill_between(time_index, prophet_lower, prophet_upper, alpha=0.3, label='95% 置信区间')
plt.title("Prophet 模型预测及95%置信区间")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.legend()
plt.grid(True)
plt.show()