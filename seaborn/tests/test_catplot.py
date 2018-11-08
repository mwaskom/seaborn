import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def test_catplot_point():
    data = np.random.rand(100, 4)
    data[np.random.randint(0, len(data), 25), 0] = np.nan
    data[:, 1] = np.tile(np.arange(1, 6), 20)
    data[:, 2] = np.repeat(np.arange(1, 11), 10)
    data[:, 3] = np.tile(np.concatenate((np.repeat(1, 5), np.repeat(2, 5))), 10)
    data = pd.DataFrame(data=data,
                        columns=['resp', 'response', 'subject', 'task'])

    try:
        sns.catplot(x='task', y='resp', hue='response', col='subject',
                    data=data, kind='point')
        plt.show()
    except ValueError as error:
        print('ValueError raised:')
        print(error)


test_catplot_point()