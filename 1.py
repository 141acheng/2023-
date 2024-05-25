import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel(r"C:\Users\lenovo\Desktop\acheng\支撑材料\数据处理结果.xlsx", usecols=["销售日期", "销量(千克)", "分类名称", "销售单价(元/千克)"])
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 使用LabelEncoder对分类名称进行编码
le = LabelEncoder()
data['Category Code'] = le.fit_transform(data['分类名称'])

# 准备保存结果的DataFrame
results = pd.DataFrame()

# 针对每个蔬菜分类进行数据分析
for category in data['分类名称'].unique():
    category_data = data[data['分类名称'] == category]

    # 使用线性回归模型拟合数据
    X = category_data[['销售单价(元/千克)']].values.reshape(-1, 1)  # Predictor
    y = category_data['销量(千克)']  # Response
    model = LinearRegression()
    model.fit(X, y)

    # 预测销量
    y_pred = model.predict(X)

    # 计算R² 和 MSE
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # 将结果存入DataFrame
    results = pd.concat([results, pd.DataFrame({
        'Category': [category],
        'R2': [r2],
        'MSE': [mse]
    })], ignore_index=True)

# 保存结果到Excel
results.to_excel(r"C:\Users\lenovo\Desktop\acheng\支撑材料\model_evaluation_results.xlsx", index=False)

# 输出结果以验证
print(results)
