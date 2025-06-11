#LSTM 模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —————— 解决中文乱码 ——————
plt.rcParams['font.sans-serif'] = ['SimHei']      # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False        # 正确显示负号

# 1. 读取并解析 CSV 数据
df = pd.read_csv(
    "D:/python/pytest/601006.csv",
    parse_dates=['Date'],
    date_parser=lambda x: pd.to_datetime(x.strip(), format='%b %d %Y'),
    index_col='Date',
    dayfirst=False
)

# 取复权收盘价
series = df[['Adj Close']]

# 2. 归一化
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series)

# 3. 构建滑动窗口序列
def create_seq(data, look_back=6):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 6
X, y = create_seq(scaled, look_back)

# 4. 划分训练/测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. 重塑为 LSTM 输入格式 [样本, 时间步, 特征]
X_train = X_train.reshape(-1, look_back, 1)
X_test  = X_test.reshape(-1, look_back, 1)

# 6. LSTM 模型构建
model_lstm = Sequential([
    LSTM(50, input_shape=(look_back, 1), return_sequences=False),
    Dropout(0.2),
    Dense(1)  # 回归输出
])
model_lstm.compile(optimizer='adam', loss='mse')

# 7. 模型训练
history = model_lstm.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=2
)

# 8. 模型预测 & 反归一化
y_pred = model_lstm.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# —————— 可视化与评估 ——————

# 9. 训练/验证损失曲线
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='训练 Loss')
plt.plot(history.history['val_loss'], label='验证 Loss')
plt.title('LSTM 训练/验证 损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# 10. 实际 vs 预测 曲线对比
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='实际收盘价', linewidth=2)
plt.plot(y_pred_actual, label='预测收盘价', linestyle='--')
plt.title('LSTM 模型：实际值 vs 预测值')
plt.xlabel('样本索引')
plt.ylabel('收盘价（元）')
plt.legend()
plt.grid(True)
plt.show()

# 11. 计算评价指标
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_actual, y_pred_actual)
r2   = r2_score(y_test_actual, y_pred_actual)
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# 12. 残差分布直方图
residuals = (y_test_actual - y_pred_actual).flatten()
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.title('LSTM 残差分布直方图')
plt.xlabel('预测误差')
plt.ylabel('频数')
plt.grid(True)
plt.show()

# 13. 实际 vs 预测 散点图
plt.figure(figsize=(6,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.6)
lims = [
    min(y_test_actual.min(), y_pred_actual.min()),
    max(y_test_actual.max(), y_pred_actual.max())
]
plt.plot(lims, lims, 'r--')
plt.title('实际值 vs 预测值 散点图')
plt.xlabel('实际收盘价')
plt.ylabel('预测收盘价')
plt.grid(True)
plt.show()