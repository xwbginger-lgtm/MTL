import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用SimHei字体显示中文（Windows常用），或DejaVu Sans作为备选
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取CSV数据集
df = pd.read_csv('医患测试数据集.csv')

# 筛选关键指标：Glucose（血糖）、BMI、HbA1c_level（糖化血红蛋白）
key_features = ['Glucose', 'BMI', 'HbA1c_level']

# 创建单个箱线图，比较患病（Diagnosis=1）和未患病（Diagnosis=0）组
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 准备数据：为每个特征创建两个组
data_to_plot = []
labels = []
for feature in key_features:
    normal = df[df['Diagnosis'] == 0][feature]
    diseased = df[df['Diagnosis'] == 1][feature]
    data_to_plot.extend([normal, diseased])
    labels.extend([f'{feature} 未患病', f'{feature} 患病'])

# 绘制箱线图
box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)

# 颜色设置：未患病蓝色，患病红色
colors = ['lightblue', 'lightcoral'] * len(key_features)
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_title('患病 vs 未患病：关键指标箱线图比较', fontsize=14)
ax.set_ylabel('数值')
ax.tick_params(axis='x', rotation=45)  # 旋转x轴标签以防重叠

plt.tight_layout()
plt.show()
