import pandas as pd
import numpy as np
import scipy.stats as stats

# 读取数据预处理的Excel文件
attachment2 = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\附件2.xlsx")

# 读取附件一的Excel文件，包含单品编码和分类编码
attachment1 = pd.read_excel(r"C:\Users\DIUDIUDIU\Desktop\CUMCM2023Problems\C题\附件1.xlsx")

# 创建单品到品类的映射表
product_category_mapping = dict(zip(attachment1["单品编码"], attachment1["分类名称"]))

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

# 步骤 2: 创建品类关联度字典
category_correlations = {}

# 步骤 3: 遍历单品关联度数据框
for index, row in sorted_correlation_df.iterrows():
    product_A = row["单品A"]
    product_B = row["单品B"]
    correlation = row["多元斯皮尔曼秩相关系数"]

    # 步骤 4: 获取单品 A 和 B 所属的品类
    category_A = product_category_mapping.get(product_A, None)
    category_B = product_category_mapping.get(product_B, None)

    # 步骤 5: 如果单品 A 和 B 都有所属品类，计算加权平均关联度
    if category_A and category_B:
        # 步骤 6: 计算权重（可以根据销售量等因素进行权重计算）
        weight_A = 1.0  # 可以根据需要进行权重计算
        weight_B = 1.0  # 可以根据需要进行权重计算
        weighted_correlation = correlation * (weight_A + weight_B) / 2.0

        # 步骤 7: 更新品类关联度字典
        if category_A not in category_correlations:
            category_correlations[category_A] = {}
        if category_B not in category_correlations:
            category_correlations[category_B] = {}

        # 步骤 8: 记录品类 A 和 B 之间的关联度
        category_correlations[category_A][category_B] = weighted_correlation
        category_correlations[category_B][category_A] = weighted_correlation

# 步骤 9: 将品类关联度字典保存为Excel文件
category_correlations_df = pd.DataFrame(category_correlations)
category_correlations_df.to_excel("品类关联度.xlsx", index=True)

