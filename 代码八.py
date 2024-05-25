import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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

# 构建多个模型，每个单品对应一个模型
unique_items = data["单品编码"].unique()
models = {}

# 划分训练集和测试集，构建并训练多个模型
for item_code in unique_items:
    item_data = data[data["单品编码"] == item_code]
    X = item_data[features].values
    y = item_data[target].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化模型
    input_dim = X.shape[1]  # 输入特征的维度
    hidden_units = 64  # 隐藏层的神经元数量
    output_dim = 1  # 输出维度，即销售量的预测

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),  # 输入层
        tf.keras.layers.Dense(hidden_units, activation='relu'),  # 隐藏层
        tf.keras.layers.Dense(output_dim)  # 输出层
    ])


    # 定义损失函数
    def custom_loss(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(y_true, y_pred)


    # 选择学习率
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # 编译模型
    model.compile(optimizer=optimizer, loss=custom_loss)

    # 迭代训练
    num_epochs = 100  # 训练轮数
    batch_size = 32  # 小批量样本大小

    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            # 获取一个小批量样本
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # 计算损失并更新参数
            with tf.GradientTape() as tape:
                predictions = model(batch_X)
                loss = custom_loss(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 计算测试集上的损失
        test_predictions = model(X_test)
        test_loss = custom_loss(y_test, test_predictions)

        # 每个epoch结束后，打印训练损失和测试损失
        print(f"Item Code: {item_code}, Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss.numpy()}")

    # 保存训练好的模型
    models[item_code] = model

# 模型训练完成，可以用于预测销售量
