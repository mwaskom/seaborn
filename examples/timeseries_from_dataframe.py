"""
Timeseries from DataFrame
=========================

"""

import seaborn as sns
sns.set(style="darkgrid")

gammas = sns.load_dataset("gammas")
sns.tsplot(gammas, "timepoint", "subject", "ROI", "BOLD signal")
