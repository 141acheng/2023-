import pandas as pd
import numpy as np
import scipy.stats as stats

# 读取数据预处理的Excel文件
attachment2 = pd.read_excel(r"C:\Users\lenovo\Desktop\acheng\支撑材料\数据处理结果.xlsx")

# 数据预处理：将销售日期转换为日期格式
attachment2["销售日期"] = pd.to_datetime(attachment2["销售日期"])

# 先确保销量列的数据都是字符串
attachment2["销量(千克)"] = attachment2["销量(千克)"].astype(str)

# 移除销量中的空格，然后将其转换为浮点数
attachment2["销量(千克)"] = attachment2["销量(千克)"].str.replace(" ", "").astype(float)

# 初始化一个空的相关系数列表
correlation_list = []

# 针对每对单品（A和B），执行以下步骤：
unique_products = attachment2["单品编码"].unique()
for i, product_A in enumerate(unique_products):
    for product_B in unique_products[i + 1:]:  # 避免重复计算
        # 获取单品A和B的销售数据
        sales_A = attachment2.loc[attachment2["单品编码"] == product_A][["销售日期", "销量(千克)"]]
        sales_B = attachment2.loc[attachment2["单品编码"] == product_B][["销售日期", "销量(千克)"]]

        # 合并单品A和B的销售数据，基于销售日期进行匹配
        merged_sales = sales_A.merge(sales_B, on="销售日期", suffixes=("_A", "_B"), how="inner")

        # 检查销售数据的标准差是否足够大，如果不够大就跳过计算
        if merged_sales["销量(千克)_A"].std() > 0.001 and merged_sales["销量(千克)_B"].std() > 0.001:
            # 执行多元斯皮尔曼秩相关系数计算
            rho, _ = stats.spearmanr(merged_sales["销量(千克)_A"], merged_sales["销量(千克)_B"])

            # 将计算得到的相关系数添加到相关系数列表中
            correlation_list.append({
                "单品A": product_A,
                "单品B": product_B,
                "多元斯皮尔曼秩相关系数": rho
            })

# 将相关系数列表转化为DataFrame格式
correlation_df = pd.DataFrame(correlation_list)

# 根据多元斯皮尔曼秩相关系数对单品之间的关联度进行排序
sorted_correlation_df = correlation_df.sort_values(by="多元斯皮尔曼秩相关系数", ascending=False)

# 保存结果为Excel文件
sorted_correlation_df.to_excel("不同单品之间的关联程度.xlsx", index=False)

