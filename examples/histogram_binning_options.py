"""
Histogram binning options on data that does not fit the bin size.
=================================================================

"""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df_round_up = pd.DataFrame({"x": list(range(58))})
    df_round_down = pd.DataFrame({"x": list(range(52))})
    n_plots = 4

    fig, ax = plt.subplots(n_plots, 2, figsize=(5 * 2, 5 * n_plots))
    fig.suptitle("Histogram binning options\nrange(58) and range(52)")
    fig.tight_layout()

    hist_n_bins_ru = sns.histplot(df_round_up,
                                  x="x", bins=10, ax=ax[0][0])
    hist_n_bins_ru.set_title("bins=10")
    hist_n_bins_rd = sns.histplot(df_round_down,
                                  x="x", bins=10, ax=ax[0][1])
    hist_n_bins_rd.set_title("bins=10")

    hist_binwidth_ru = sns.histplot(df_round_up,
                                    x="x", binwidth=10, ax=ax[1][0])
    hist_binwidth_ru.set_title("binwidth=10")
    hist_binwidth_rd = sns.histplot(df_round_down,
                                    x="x", binwidth=10, ax=ax[1][1])
    hist_binwidth_rd.set_title("binwidth=10")

    hist_binrange_ru = sns.histplot(df_round_up,
                                    x="x", binrange=(0, 60), ax=ax[2][0])
    hist_binrange_ru.set_title("binrange=(0, 60)")
    hist_binrange_rd = sns.histplot(df_round_down,
                                    x="x", binrange=(0, 60), ax=ax[2][1])
    hist_binrange_rd.set_title("binrange=(0, 60)")

    hist_binrange_and_width_ru = sns.histplot(df_round_up,
                                              x="x", binwidth=10,
                                              binrange=(0, 60), ax=ax[3][0])
    hist_binrange_and_width_ru.set_title("bw=10, br=(0, 60)")
    hist_binrange_and_width_rd = sns.histplot(df_round_down,
                                              x="x", binwidth=10,
                                              binrange=(0, 60), ax=ax[3][1])
    hist_binrange_and_width_rd.set_title("bw=10, br=(0, 60)")

    plt.savefig("output_files/binning_options.svg")
