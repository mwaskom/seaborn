import matplotlib as mpl


def MarkerStyle(marker=None, fillstyle=None):
    """
    Allow MarkerStyle to accept a MarkerStyle object as parameter.

    Supports matplotlib < 3.3.0
    https://github.com/matplotlib/matplotlib/pull/16692

    """
    if isinstance(marker, mpl.markers.MarkerStyle):
        if fillstyle is None:
            return marker
        else:
            marker = marker.get_marker()
    return mpl.markers.MarkerStyle(marker, fillstyle)
