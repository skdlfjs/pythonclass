import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
#读取数据
import pandas as pd

df = pd.read_csv(r"D:\python\pytest\601006.csv")

# —————— 解决中文乱码 ——————
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取并解析 CSV（确保 Date 列已转为 datetime）
df = pd.read_csv(
    "D:/python/pytest/601006.csv",
    parse_dates=['Date'],
    date_parser=lambda x: pd.to_datetime(x.strip(), format='%b %d %Y'),
    dayfirst=False
)
df.set_index('Date', inplace=True)

# 2. 准备 Prophet 所需数据框
df_prophet = df[['Adj Close']].reset_index().rename(columns={'Date': 'ds', 'Adj Close': 'y'})
# 确保 ds 列为 datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# 3. 定义节假日示例
holidays = pd.DataFrame({
    'holiday': 'Spring_Festival',
    'ds': pd.to_datetime(['2019-02-04','2020-01-24','2021-02-11','2022-01-31','2023-01-21']),
    'lower_window': 0,
    'upper_window': 3,
})

# 4. 构建并训练模型
model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    holidays=holidays,
    seasonality_prior_scale=10
)
model_prophet.fit(df_prophet)  # 这里不会再报 ds 解析错误

# 5. 生成未来 12 周的日期框架并预测
future = model_prophet.make_future_dataframe(periods=12, freq='W')
forecast_prophet = model_prophet.predict(future)

# 6. 可视化：历史拟合 + 未来预测
plt.figure(figsize=(12,6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='历史真实值')
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], 
         label='Prophet 拟合/预测', linestyle='--')
# 突出未来部分
train_end = df_prophet['ds'].max()
future_mask = forecast_prophet['ds'] > train_end
plt.plot(forecast_prophet.loc[future_mask, 'ds'],
         forecast_prophet.loc[future_mask, 'yhat'],
         label='未来12周预测', linestyle=':', linewidth=2)
plt.title('Prophet 模型：历史拟合与未来12周预测')
plt.xlabel('日期')
plt.ylabel('复权收盘价')
plt.legend()
plt.grid(True)
plt.show()

# 7. 组件分解图
model_prophet.plot_components(forecast_prophet)
plt.show()

# 8. 误差评估（如果有实际后12周数据可比较；此处示例使用最后12周历史进行演示）
actual = df_prophet['y'].iloc[-12:].values
pred   = forecast_prophet['yhat'].iloc[-12:].values

mse  = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(actual, pred)

print(f"Prophet 模型 未来12周 演示评估：RMSE={rmse:.4f}, MAE={mae:.4f}")
