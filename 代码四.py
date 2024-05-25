import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体（SimHei）来显示中文


# 读取数据
data = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附录\数据处理结果.xlsx", usecols=["销售日期", "销量(千克)", "分类名称", "销售单价(元/千克)"])
data["销售日期"] = pd.to_datetime(data["销售日期"])

# 使用LabelEncoder对分类名称进行编码
le = LabelEncoder()
data["分类编码"] = le.fit_transform(data["分类名称"])

# 标准化销量和定价数据
scaler = StandardScaler()
data[["销量(千克)", "销售单价(元/千克)"]] = scaler.fit_transform(data[["销量(千克)", "销售单价(元/千克)"]])

# 加载已训练的模型
loaded_model = load_model("sales_prediction_model_with_category.h5")

# 用户输入分类名称和销售单价
user_categories = ["辣椒类", "茄类", "花叶类", "食用菌", "花菜类", "水生根茎类"]  # 请替换为用户输入的分类名称列表
user_cost_markup_range = list(range(1, 16))  # 请替换为用户输入的销售单价范围

# 创建DataFrame用于保存结果
results_df = pd.DataFrame(columns=user_categories, index=user_cost_markup_range)

# 构建预测输入
for category in user_categories:
    user_category_encoded = le.transform([category])
    X_user_category = np.array(user_category_encoded).reshape(1, -1)
    for user_cost_markup in user_cost_markup_range:
        X_user_cost_markup = np.array([[user_cost_markup]])  # 只提供销售单价

        # 进行销量预测
        predicted_sales = loaded_model.predict([X_user_cost_markup, X_user_category])

        # 构建反标准化模型
        input_layer = Input(shape=(1,))
        output_layer = Dense(2)(input_layer)  # 2表示输出2个神经元，匹配scaler.inverse_transform期望的形状
        inverse_scaler_model = Model(inputs=input_layer, outputs=output_layer)

        # 对预测结果进行反标准化
        predicted_sales = inverse_scaler_model.predict(predicted_sales)

        # 保存结果
        results_df.loc[user_cost_markup, category] = predicted_sales[0][0]

# 输出结果为Excel文件
results_df.to_excel("predicted_sales_results.xlsx")

# 绘制散点折线图
font_path = "path/to/your/font.ttf"  # 设置字体文件路径，替换为您的字体文件路径
font_prop = FontProperties(fname=font_path)

plt.figure(figsize=(10, 6))
for category in user_categories:
    plt.plot(results_df.index, results_df[category], marker='o', label=category)

plt.xlabel("定价", fontproperties=font_prop)  # 指定字体属性
plt.ylabel("销量", fontproperties=font_prop)  # 指定字体属性
plt.title("不同品类的销量随定价的变化", fontproperties=font_prop)  # 指定字体属性
plt.legend(prop=font_prop)  # 指定字体属性
plt.grid(True)

# 保存图表为图片文件
plt.savefig("sales_vs_price.png")

# 显示图表
plt.show()

# 输出预测结果
print("预测结果已保存为 predicted_sales_results.xlsx 文件。")
print("散点折线图已保存为 sales_vs_price.png 文件。")