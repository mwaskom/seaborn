import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
import seaborn as sns

sns.set()

x = np.linspace(0, 5, 100)
for n in range(1, 6):
    plt.plot(x, jn(x, n))
plt.title("Default Seaborn Aesthetics")
plt.savefig("examples/example_plot.png")
