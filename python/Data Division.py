#读取数据
import pandas as pd

df = pd.read_csv(r"D:\python\pytest\601006.csv")

#特征选择与归一化
from sklearn.preprocessing import MinMaxScaler

# 只保留目标字段
close_data = df[['Adj Close']].copy()

# 归一化处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# 重新转为 DataFrame
import numpy as np
scaled_df = pd.DataFrame(scaled_data, index=close_data.index, columns=["Adj Close"])

#构建训练集与测试集
# 数据划分
train_size = int(len(scaled_df) * 0.8)
train_data = scaled_df.iloc[:train_size]
test_data = scaled_df.iloc[train_size:]
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# 创建 LSTM 输入样本
time_step = 10
X_train, y_train = create_dataset(train_data.values, time_step)
X_test, y_test = create_dataset(test_data.values, time_step)
# reshape 为 [样本数, 时间步长, 特征数]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)