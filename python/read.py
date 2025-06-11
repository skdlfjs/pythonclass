#读取数据
import pandas as pd

df = pd.read_csv(r"D:\python\pytest\601006.csv")
print(df.head())

#缺失值处理
df_clean = df.copy()
df_clean = df_clean.ffill()  # 前向填充缺失值

#异常值处理
# 以收盘价为例，去除超出均值±3σ的数据
import numpy as np
series = df_clean['Close']
mean = series.mean()
std = series.std()
df_clean['Close'] = np.where(series > mean + 3*std, mean + 3*std,
                              np.where(series < mean - 3*std, mean - 3*std, series))

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