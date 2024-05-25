import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\lenovo\Desktop\acheng\支撑材料\数据处理结果.xlsx", usecols=["销售日期", "销量(千克)", "分类名称", "销售单价(元/千克)"])

# data中包含"销售单价(元/千克)"（作为特征）和"销量是否达标"（作为目标）

# 假设我们使用0.5的销量作为阈值来定义分类标签
data['Is_High_Sale'] = (data['销量(千克)'] > 0.5).astype(int)

# 特征和标签
X = data[['销售单价(元/千克)']]
y = data['Is_High_Sale']

# 拆分数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归模型并训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测测试集的概率
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# 计算AUC值
auc_score = roc_auc_score(y_test, y_pred_prob)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
