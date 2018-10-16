'''
This file is an example for pairing the line plot with heatplot.
'''

from matplotlib import gridspec
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy


def drawHeat_plot(line, data, xSize, ySize, xtick, tick_spacing, cols='sin',
                filename='HeatPlot', title='Demo',
                yticklabel=['Value']):  
    textFS = 80
    titleFS = 100
    LenData = len(line)
    rc = {'font.size': textFS, 'xtick.labelsize': textFS, 'ytick.labelsize': textFS, }
    sns.set(rc=rc)
    size = np.shape(data)
    fig = plt.figure(figsize=(xSize, ySize))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[1])

    heatplt = sns.heatmap(data, linewidths=0, ax=ax1, cmap="YlGnBu", annot=False,
                           xticklabels=xtick,
                          cbar=False, yticklabels=yticklabel)

    plt.xticks(fontsize=textFS)
    plt.yticks(fontsize=textFS)
    ax1.set_ylabel('Value', fontsize=textFS)
    ax1.set_xlabel('Time step', fontsize=textFS)
    plt.xlim([0, LenData])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(line, linewidth=3)
    ax0.set_ylabel('Value', fontsize=textFS)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax0.set_title(title, fontsize=titleFS)
    ax0.grid(linestyle='--', linewidth='0.5', color='black')
    plt.legend([cols + ' Data'], fontsize=textFS)
    plt.xlim([0, LenData])

    fig = heatplt.get_figure()
    fig.savefig(filename + ".png", format='png', bbox_inches='tight', transparent=True)
    
line = np.sin([3.14*i/20 for i in range(0,100)])
data = np.sin([3.14*i/20 for i in range(0,100)])
line = np.reshape(line,[100,])
data = np.reshape(data,[1,100])
xSize = 150
ySize = 30
SliceStart = 0
SliceEnd = 100
TickSpace = 10

drawHeat_plot(line, data, xSize, ySize, range(SliceStart - TickSpace, SliceEnd, TickSpace), tick_spacing=TickSpace, cols='sin',
                filename='LineHeatPlot', title='Demo')
                
