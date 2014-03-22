from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def palplot(pal, size=1):
    """Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot

    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def puppyplot(grown_up=False):
    """Plot today's daily puppy. Only works in the IPython notebook."""
    from .external.six.moves.urllib.request import urlopen
    from IPython.display import HTML
    try:
        from bs4 import BeautifulSoup
        url = "http://www.dailypuppy.com"
        if grown_up:
            url += "/dogs"
        html_doc = urlopen(url)
        soup = BeautifulSoup(html_doc)
        puppy = soup.find("div", {"class": "daily_puppy"})
        return HTML(str(puppy.img))
    except ImportError:
        html = ('<img  src="http://cdn-www.dailypuppy.com/dog-images/'
                'decker-the-nova-scotia-duck-tolling-retriever_'
                '72926_2013-11-04_w450.jpg" style="width:450px;"/>')
        return HTML(html)
