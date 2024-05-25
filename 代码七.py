import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取数据
data = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附录\数据处理结果.xlsx",
                     usecols=["销售日期", "销量(千克)", "分类名称"])
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

# 加载训练好的RNN模型
trained_model = load_model("sales_prediction_model_with_category.h5")

# 定义损失率和批发价格
category_loss_constraints = {
    "水生根茎类": 9.46,
    "花叶类": 9.99,
    "花菜类": 9.57,
    "茄类": 9.52,
    "辣椒类": 9.44,
    "食用菌": 9.55
}

wholesale_prices = {
    "水生根茎类": 8.085504,
    "花叶类": 4.446458,
    "花菜类": 5.771536,
    "茄类": 5.689737,
    "辣椒类": 7.692309,
    "食用菌": 6.146643
}

# 定义目标函数
def objective_function(prices):
    # 计算总收益
    total_revenue = 0
    total_loss = 0

    # 批处理优化
    batch_size = 10000
    for start in range(0, len(X_sales_date), batch_size):
        end = start + batch_size
        batch_X_sales_date = X_sales_date[start:end]
        batch_X_category = X_category[start:end]

        sales_predictions = trained_model.predict([batch_X_sales_date, batch_X_category])
        total_revenue += np.sum(sales_predictions * prices)

        # 计算总损耗
        total_loss += np.sum([category_loss_constraints[le.inverse_transform([category])[0]] * np.sum(sales_predictions[batch_X_category == category]) for category in np.unique(batch_X_category)])

    # 最大化总收益，同时满足损耗约束
    return -(total_revenue - total_loss)

# 构建初始价格向量
initial_prices = np.array([wholesale_prices[le.inverse_transform([category])[0]] for category in X_category])

# 定义约束条件列表
def category_loss_constraint(prices, category):
    # 批处理优化
    total_loss = 0
    batch_size = 10000
    for start in range(0, len(X_sales_date), batch_size):
        end = start + batch_size
        batch_X_sales_date = X_sales_date[start:end]
        batch_X_category = X_category[start:end]

        # 计算该品类的销售量预测
        batch_sales_predictions = trained_model.predict([batch_X_sales_date, np.array([category])])

        # 计算该品类的总损耗
        total_loss += np.sum(category_loss_constraints[le.inverse_transform([category])[0]] * np.sum(batch_sales_predictions) - np.sum(batch_sales_predictions))

    return total_loss

constraints = [{'type': 'ineq', 'fun': category_loss_constraint, 'args': (category,)} for category in np.unique(X_category)]

# 最大化总收益，同时满足约束条件
result = minimize(objective_function, initial_prices, constraints=constraints, method='SLSQP')

# 输出最优的定价策略
optimal_prices = result.x

# 输出结果
for category, price in zip(np.unique(X_category), optimal_prices):
    print(f"Category: {le.inverse_transform([category])[0]}, Optimal Price: {price}")