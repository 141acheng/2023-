import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam

# 读取数据
data = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附录\数据处理结果.xlsx", usecols=["销售日期", "销量(千克)", "分类名称"])
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 使用LabelEncoder对分类名称进行编码
le = LabelEncoder()
data["分类编码"] = le.fit_transform(data["分类名称"])

# 标准化销量数据
scaler = StandardScaler()
data["销量(千克)"] = scaler.fit_transform(data[["销量(千克)"]])

# 数据准备
X_sales_date = data["销售日期"].values.astype(np.int64) // 10 ** 9  # 转换为Unix时间戳
X_category = data["分类编码"].values
y = data["销量(千克)"].values

X_train_sales_date, X_test_sales_date, X_train_category, X_test_category, y_train, y_test = train_test_split(
    X_sales_date, X_category, y, test_size=0.2, random_state=42)

# 构建RNN模型
input_sales_date = Input(shape=(1,), name="sales_date_input")
input_category = Input(shape=(1,), name="category_input")

embedding_layer = Embedding(input_dim=len(np.unique(X_category)), output_dim=10)(input_category)
flatten_layer = Flatten()(embedding_layer)

concat_layer = Concatenate()([input_sales_date, flatten_layer])
dense_layer = Dense(64, activation='relu')(concat_layer)
output_layer = Dense(1)(dense_layer)

model = tf.keras.Model(inputs=[input_sales_date, input_category], outputs=output_layer)

# 更改优化器为Adam，尝试不同的学习率
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))  # 尝试不同的学习率值

# 模型训练
model.fit([X_train_sales_date, X_train_category], y_train, epochs=10, batch_size=64)  # 增加批量大小

# 模型性能评估
y_pred = model.predict([X_test_sales_date, X_test_category])
mse = mean_squared_error(y_test, y_pred)
print("均方误差 (MSE):", mse)

# 保存模型
model.save("sales_prediction_model_with_category.h5")
