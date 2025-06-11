import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ——————— 数据读取 & 预处理 ———————
df = pd.read_csv(r"D:\python\pytest\601006.csv",
                 parse_dates=['Date'],
                 date_parser=lambda x: pd.to_datetime(x.strip(), format='%b %d %Y'),
                 index_col='Date')
df.index = pd.to_datetime(df.index).tz_localize(None)
series = df['Adj Close'].dropna()


# 1. 指定默认字体为黑体（SimHei），确保已安装该字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ——————— 1. ARIMA 模型 ———————
# 1.1 差分平稳
if adfuller(series)[1] > 0.05:
    series_diff = series.diff().dropna()
else:
    series_diff = series

# 1.2 ACF/PACF 可视化（可选）
# fig, axes = plt.subplots(2,1)
# sm.graphics.tsa.plot_acf(series_diff, lags=20, ax=axes[0])
# sm.graphics.tsa.plot_pacf(series_diff, lags=20, ax=axes[1])
# plt.show()

# 1.3 网格搜索最优 (p,d,q)
best_aic, best_order = np.inf, None
for p in range(4):
    for q in range(4):
        try:
            m = ARIMA(series, order=(p,1,q)).fit()
            if m.aic < best_aic:
                best_aic, best_order = m.aic, (p,1,q)
        except:
            pass
print("ARIMA最优参数:", best_order)

# 1.4 拟合与预测
model_arima = ARIMA(series, order=best_order).fit()
# 训练集拟合（可选）
# fitted = model_arima.fittedvalues

# 未来12周预测
forecast_arima = model_arima.forecast(steps=12)
last_date = series.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                               periods=12, freq='W')
forecast_series = pd.Series(forecast_arima.values, index=forecast_index)

# ——————— 2. Prophet 模型 ———————
df_prophet = series.reset_index().rename(columns={'Date':'ds','Adj Close':'y'})
m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False)
m_prophet.fit(df_prophet)
future = m_prophet.make_future_dataframe(periods=12, freq='W')
forecast_prophet = m_prophet.predict(future)

# 提取最后12周预测
mask = forecast_prophet['ds'] > df_prophet['ds'].max()
future_prophet = forecast_prophet.loc[mask].iloc[:12]
prophet_pred = future_prophet['yhat'].values

# ——————— 3. LSTM 模型 ———————
# 3.1 归一化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.values.reshape(-1,1))

# 3.2 滑动窗口
def create_seq(data, lb=6):
    X,y = [],[]
    for i in range(lb, len(data)):
        X.append(data[i-lb:i,0]); y.append(data[i,0])
    return np.array(X), np.array(y)

look_back = 6
X, y = create_seq(scaled, look_back)
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = X_train.reshape(-1,look_back,1)
X_test  = X_test.reshape(-1,look_back,1)

# 3.3 构建与训练
model_lstm = Sequential([ LSTM(50,input_shape=(look_back,1)), Dropout(0.2), Dense(1) ])
model_lstm.compile('adam','mse')
model_lstm.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=0)

# 3.4 预测 & 反归一化
y_pred = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(y_pred).flatten()
lstm_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

# 取最后12周
lstm_pred = lstm_pred[-12:]
lstm_true = lstm_true[-12:]

# ——————— 4. 评估指标 & 可视化 ———————
# 4.1 定义评估函数
def evaluate(a, p):
    return (np.sqrt(mean_squared_error(a,p)), mean_absolute_error(a,p))

# 4.2 计算指标
# ARIMA
arima_true = series.iloc[-12:].values
arima_pred = forecast_series.values
rmse_a, mae_a = evaluate(arima_true, arima_pred)
# Prophet
prophet_true = series.iloc[-12:].values
rmse_p, mae_p = evaluate(prophet_true, prophet_pred)
# LSTM
rmse_l, mae_l = evaluate(lstm_true, lstm_pred)

print(f"ARIMA   RMSE={rmse_a:.4f}, MAE={mae_a:.4f}")
print(f"Prophet RMSE={rmse_p:.4f}, MAE={mae_p:.4f}")
print(f"LSTM    RMSE={rmse_l:.4f}, MAE={mae_l:.4f}")

# 4.3 条形图对比
models = ['ARIMA','Prophet','LSTM']
rmses  = [rmse_a, rmse_p, rmse_l]
maes   = [mae_a, mae_p, mae_l]
x = np.arange(3); width=0.35
plt.figure(figsize=(8,4))
plt.bar(x-width/2, rmses, width, label='RMSE')
plt.bar(x+width/2, maes, width, label='MAE')
plt.xticks(x, models); plt.ylabel('误差值'); plt.title('模型误差对比')
plt.legend(); plt.grid(axis='y',linestyle='--',alpha=0.7); plt.show()


# 4.4 散点图：实际值 vs 预测值（3 子图并排）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, name, actual, pred in zip(
    axes,
    models,
    [arima_true, prophet_true, lstm_true],
    [arima_pred, prophet_pred, lstm_pred]
):
    ax.scatter(actual, pred, alpha=0.6)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', label='理想 $y=x$')
    ax.set_title(f'{name}：实际 vs 预测')
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# 4.5 残差直方图（3 子图并排）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, name, actual, pred in zip(
    axes,
    models,
    [arima_true, prophet_true, lstm_true],
    [arima_pred, prophet_pred, lstm_pred]
):
    resid = actual - pred
    ax.hist(resid, bins=20, edgecolor='k', alpha=0.7)
    ax.set_title(f'{name} 残差分布')
    ax.set_xlabel('残差 = 实际 - 预测')
    ax.set_ylabel('频数')
    ax.grid(True)

plt.tight_layout()
plt.show()
