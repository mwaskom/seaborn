"""
Linear regression with marginal distributions
=============================================

"""
import seaborn as sns
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
color = sns.color_palette()[2]
g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",
                  color=color, size=7)
g.ax_joint.set(xlim=(0, 70), ylim=(0, 12))
