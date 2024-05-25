import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from geneticalgorithm import geneticalgorithm as ga  # 需要安装 geneticalgorithm 库

# 读取Excel数据
data = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附录\数据处理结果.xlsx")

# 选择需要的特征列
selected_features = ["销售日期", "销量(千克)", "销售单价(元/千克)", "单品编码", "是否打折销售"]
data = data[selected_features]

# 将销售日期列转换为日期时间类型
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 使用LabelEncoder对单品编码进行编码
le = LabelEncoder()
data["单品编码"] = le.fit_transform(data["单品编码"])

# 选择用于模型训练的特征和目标列
features = ["销售单价(元/千克)", "销售日期", "单品编码", "是否打折销售"]
target = ["销量(千克)"]

# 标准化特征数据
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# 构建回归模型（与上一段代码相同）
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # 输入层
    tf.keras.layers.LSTM(64, activation='relu'),  # LSTM隐藏层
    tf.keras.layers.Dense(1)  # 输出层
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

# 训练模型
num_epochs = 100  # 训练轮数
batch_size = 32  # 小批量样本大小

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_X)
            loss = custom_loss(batch_y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # 每个epoch结束后，打印训练损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.numpy()}")

# 使用遗传算法进行参数优化
def fitness(params):
    # 解码参数
    lstm_units, learning_rate = params

    # 构建模型
    genetic_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.LSTM(lstm_units, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # 编译模型
    genetic_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=custom_loss)

    # 训练模型
    num_epochs = 100  # 训练轮数
    batch_size = 32  # 小批量样本大小

    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                predictions = genetic_model(batch_X)
                loss = custom_loss(batch_y, predictions)
            gradients = tape.gradient(loss, genetic_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, genetic_model.trainable_variables))

    # 预测销售量
    y_pred = genetic_model.predict(X_test)

    # 计算模型性能指标
    mse = mean_squared_error(y_test, y_pred)

    return mse

# 定义参数范围和遗传算法参数
varbound = np.array([[32, 128], [0.0001, 0.01]])  # 参数范围
algorithm_param = {'max_num_iteration': 100, 'population_size': 10}  # 遗传算法参数

# 使用遗传算法求解最佳参数
model_genetic = ga(function=fitness, dimension=2, variable_type='real', variable_boundaries=varbound,
                    algorithm_parameters=algorithm_param)

# 运行遗传算法
model_genetic.run()

# 获取最佳参数
best_params = model_genetic.output_dict['variable']

# 输出最佳参数
print("Best Parameters (LSTM Units, Learning Rate):", best_params)
