#ARIMA 模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt
#读取数据
import pandas as pd

df = pd.read_csv(r"D:\python\pytest\601006.csv")

# 1. 指定默认字体为黑体（SimHei），确保已安装该字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取 CSV 并解析日期
df = pd.read_csv(
    "D:/python/pytest/601006.csv", 
    parse_dates=['Date'], 
    date_parser=lambda x: pd.to_datetime(x.strip(), format='%b %d %Y'),
    dayfirst=False
)
df.set_index('Date', inplace=True)

# 确保索引为纯 datetime，无时区
df.index = pd.to_datetime(df.index).tz_localize(None)

# 取出复权收盘价序列
series = df['Adj Close']

# 2. 平稳性检验
adf_p = adfuller(series)[1]
print(f"原序列 ADF p-value = {adf_p:.4f}")

# 3. 差分并检验
series_diff = series.diff().dropna()
adf_p2 = adfuller(series_diff)[1]
print(f"差分一次后 ADF p-value = {adf_p2:.4f}")

# 4. ACF/PACF 图
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
sm.graphics.tsa.plot_acf(series_diff, lags=20, ax=axes[0])
sm.graphics.tsa.plot_pacf(series_diff, lags=20, ax=axes[1])
plt.tight_layout()
plt.show()

# 5. 网格搜索最优 (p,d,q)
best_aic = np.inf
best_order = None
for p in range(4):
    for q in range(4):
        try:
            m = ARIMA(series, order=(p, 1, q)).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_order = (p, 1, q)
        except:
            continue
print(f"最优 ARIMA 参数: p,d,q = {best_order}, AIC = {best_aic:.2f}")

# 6. 拟合模型并输出摘要
model_arima = ARIMA(series, order=best_order).fit()
print(model_arima.summary())

# 7. 拟合值 vs 实际值（训练集部分对比）
#    start=series.index[1] 是因为 d=1 差分后首个可预测点从第二行开始
fitted = model_arima.predict(start=series.index[1], end=series.index[-1])
plt.figure(figsize=(12,6))
plt.plot(series, label='原始数据')
plt.plot(fitted, label='ARIMA 拟合值', linestyle='--')
plt.title("ARIMA 模型拟合效果")
plt.legend()
plt.show()

# 8. 未来 12 周 预测
last_date = series.index[-1]
# 按周生成未来 12 周的日期
forecast_index = [last_date + pd.Timedelta(weeks=i) for i in range(1, 13)]
forecast = model_arima.forecast(steps=12)
forecast_series = pd.Series(forecast.values, index=forecast_index)

plt.figure(figsize=(12,6))
plt.plot(series, label='历史数据')
plt.plot(forecast_series, label='未来12周预测', linestyle='--', color='orange')
plt.title("ARIMA 模型未来12周预测")
plt.legend()
plt.show()

# 9. 残差分析
residuals = model_arima.resid
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(residuals)
plt.title("残差时序图")

plt.subplot(1,2,2)
sm.qqplot(residuals, line='s', ax=plt.gca())
plt.title("残差 Q-Q 图")

plt.tight_layout()
plt.show()
