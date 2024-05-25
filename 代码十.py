import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取Excel数据
data = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附录\数据处理结果.xlsx")

# 选择需要的特征列
selected_features = ["销售日期", "销量(千克)", "销售单价(元/千克)", "单品编码"]
data = data[selected_features]

# 将销售日期列转换为日期时间类型
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 使用LabelEncoder对单品编码进行编码
le = LabelEncoder()
data["单品编码"] = le.fit_transform(data["单品编码"])

# 选择用于模型训练的特征和目标列
features = ["销量(千克)", "销售单价(元/千克)", "销售日期", "单品编码"]
target = ["销量(千克)"]

# 标准化特征数据
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 加载预训练模型
pretrained_model = tf.keras.models.load_model('classification_model.h5')

# 选择测试集数据
X_test = data[features].values
y_test = data[target].values

# 进行模型预测
y_pred = pretrained_model.predict(X_test)

# 计算模型性能指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印性能指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
