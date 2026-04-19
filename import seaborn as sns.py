import seaborn as sns
import pandas as pd
import numpy as np

# 创建测试数据
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
})

# 使用 GridLayout
layout = sns.GridLayout(2, 2, height=3)
layout.add_plot(0, 0, sns.scatterplot, data=df, x='x', y='y', hue='category')
layout.add_plot(0, 1, sns.histplot, data=df, x='x')
layout.add_plot(1, 0, sns.boxplot, data=df, x='category', y='y')
layout.add_plot(1, 1, sns.lineplot, data=df, x='x', y='y')
fig = layout.render()

# 使用便捷函数
fig = sns.plot_grid([
    {'plot_func': sns.scatterplot, 'data': df, 'x': 'x', 'y': 'y'},
    {'plot_func': sns.histplot, 'data': df, 'x': 'x'},
], nrows=1, ncols=2)