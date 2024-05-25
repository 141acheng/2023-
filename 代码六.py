import pandas as pd

# 读取附件一数据
data_categories = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\CUMCM2023Problems\C题\附件1.xlsx", usecols=["单品编码", "分类名称"])

# 读取附件四数据，指定"A"列和"C"列作为需要的列
data_loss_rate = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附件4.xlsx", usecols=["A", "C"])

# 重命名列名为 "单品编码" 和 "损耗率(%)"
data_loss_rate.columns = ["单品编码", "损耗率(%)"]

# 合并两个数据集
merged_data = pd.merge(data_categories, data_loss_rate, on="单品编码")

# 计算每个品类的总销售量和总损耗率
category_loss_rate = merged_data.groupby("分类名称")["损耗率(%)"].mean()

# 打印结果
print(category_loss_rate)

