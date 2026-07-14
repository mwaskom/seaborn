import pandas as pd
import seaborn as sns

empty_df = pd.DataFrame(columns=["x", "y", "style"])

sns.relplot(data=empty_df, x="x", y="y", style="style")