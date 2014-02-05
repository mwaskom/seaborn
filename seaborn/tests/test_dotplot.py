import sys
sys.path = [x for x in sys.path if "seaborn" not in x]
sys.path.append("/afs/umich.edu/user/k/s/kshedden/fork-seaborn/seaborn")


import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

pdf = PdfPages("test_dotplot.pdf")

plt.clf()

# simple dotplot with 1 point per line
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X")
plt.title("Simple dotplot with 1 point per line, default style")
pdf.savefig()


# simple dotplot with 1 point per line
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X", striped=True)
plt.title("Simple dotplot with 1 point per line, default style with stripes")
plt.xlim(-5, 5)
pdf.savefig()

# simple dotplot with 1 point per line
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X", horizontal=False)
plt.title("Simple dotplot with 1 point per line, default vertical style")
pdf.savefig()

# simple dotplot with 1 point per line
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X", striped=True, gridlines=True, horizontal=False)
plt.title("Simple dotplot with 1 point per line, default vertical style with stripes")
plt.xlim(-5, 5)
pdf.savefig()

# simple dotplot with 1 point per line
sns.set(style="whitegrid")
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X")
plt.title("Simple dotplot with 1 point per line, whitegrid style")
pdf.savefig()

# simple dotplot with 1 point per line
sns.set(style="ticks")
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X")
plt.title("Simple dotplot with 1 point per line, ticks style")
pdf.savefig()

# simple dotplot with 1 point per line
sns.set(style="nogrid")
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X")
plt.title("Simple dotplot with 1 point per line, nogrid style")
pdf.savefig()

# simple dotplot with 1 point per line
sns.set(style="nogrid")
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
sns.dotplot(DF, points="X", striped=True)
plt.title("Simple dotplot with 1 point per line,\nnogrid style with stripes")
plt.xlim(-5, 5)
pdf.savefig()

# Dotplot with 1 point and an interval per line
sns.set()
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
DF["Y"] = np.ones(20)
sns.dotplot(DF, points="X", intervals="Y")
plt.title("Symmetric intervals")
pdf.savefig()

# Dotplot with 1 point and a nonsymmetric interval per line
plt.clf()
DF = pd.DataFrame(index=range(20))
DF["X"] = np.random.normal(size=20)
DF["Y"] = np.ones(20)
DF["Z"] = 3*np.ones(20)
sns.dotplot(DF, points="X", intervals=("Y","Z"))
plt.title("Nonsymmetric intervals")
pdf.savefig()

# Several points per line, no intervals
plt.clf()
DF = pd.DataFrame(index=range(60))
DF["X"] = np.random.normal(size=60)
DF["G"] = ["ABCDEFGHIJ"[int(k)] for k in
           np.kron(range(10), np.ones(6))]
sns.dotplot(DF, points="X", lines="G")
plt.title("10 lines, 6 points per line")
pdf.savefig()

# Several points per line, no intervals, split the labels
plt.clf()
DF = pd.DataFrame(index=range(60))
DF["X"] = np.random.normal(size=60)
DF["G"] = ["ABCDEFGHIJ"[int(k)] + "::" + str(int(k+1)) for k in
           np.kron(range(10), np.ones(6))]
sns.dotplot(DF, points="X", lines="G", split_names="::")
plt.title("10 lines, 6 points per line, split labels")
pdf.savefig()

# several points per line and a legend
plt.clf()
DF = pd.DataFrame(index=range(60))
DF["X"] = np.random.normal(size=60)
DF["G"] = ["ABCDEFGHIJ"[int(k)] for k in
           np.kron(range(10), np.ones(6))]
DF["S"] = np.kron(np.ones(10), range(6)).astype(np.int32)
ax = sns.dotplot(DF, points="X", lines="G", styles="S", striped=True)
handles, labels = ax.get_legend_handles_labels()
ii = np.argsort([float(i) for i in labels])
plt.legend([handles[i] for i in ii], [labels[i] for i in ii])
plt.title("Several points per line and a legend")
pdf.savefig()

# 2 points with intervals per line
plt.clf()
DF = pd.DataFrame(index=range(40))
DF["X"] = np.random.normal(size=40)
DF["G"] = np.kron(range(20), np.ones(2)).astype(np.int32)
DF["S"] = np.kron(np.ones(20), range(2)).astype(np.int32)
DF["I"] = np.random.uniform(1, 3, 40)
ax = sns.dotplot(DF, points="X", lines="G", styles="S",
                 intervals="I", striped=True)
plt.title("Two points and two symmetric intervals per line")
plt.xlim(-6, 6)
pdf.savefig()

# 2 points with intervals per line, 2 sections
plt.clf()
DF = pd.DataFrame(index=range(40))
DF["X"] = np.random.normal(size=40)
DF["G"] = np.kron(range(20), np.ones(2)).astype(np.int32)
DF["S"] = np.kron(np.ones(20), range(2)).astype(np.int32)
DF["I"] = np.random.uniform(1, 3, 40)
DF["T"] = np.kron((0,1), np.ones(20))
ax = sns.dotplot(DF, points="X", lines="G", sections="T", styles="S",
             intervals="I", striped=True)
plt.title("Two points and two symmetric intervals per line, 2 sections")
plt.xlim(-6, 6)
pdf.savefig()


# 2 points with intervals per line, 2 sections, vertical
plt.clf()
ax = plt.axes([0.1, 0.1, 0.8, 0.7])
DF = pd.DataFrame(index=range(40))
DF["X"] = np.random.normal(size=40)
DF["G"] = np.kron(range(20), np.ones(2)).astype(np.int32)
DF["S"] = np.kron(np.ones(20), range(2)).astype(np.int32)
DF["I"] = np.random.uniform(1, 3, 40)
DF["T"] = np.kron((0,1), np.ones(20))
ax = sns.dotplot(DF, points="X", lines="G", sections="T", styles="S",
                 intervals="I", striped=True, stacked=True, ax=ax,
                 horizontal=False)
txt = plt.title("Two points and two symmetric intervals per line, 2 sections")
txt.set_position((0.5, 1.05))
plt.xlim(-6, 6)
plt.ylim(-5, 5)
pdf.savefig()


pdf.close()
