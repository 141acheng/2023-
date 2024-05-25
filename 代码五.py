import pandas as pd

# 读取附件三数据
data_cost = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\CUMCM2023Problems\C题\附件3.xlsx", usecols=["单品编码", "批发价格(元/千克)"])

# 读取附件一数据
data_categories = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\CUMCM2023Problems\C题\附件1.xlsx", usecols=["单品编码", "分类名称"])

# 合并两个数据集
merged_data = pd.merge(data_cost, data_categories, on="单品编码")

# 计算不同品类的平均成本
average_cost_by_category = merged_data.groupby("分类名称")["批发价格(元/千克)"].mean()

# 打印结果
print(average_cost_by_category)
