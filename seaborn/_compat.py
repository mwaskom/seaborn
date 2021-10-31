from distutils.version import LooseVersion
import numpy as np
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


def norm_from_scale(scale, norm):
    """Produce a Normalize object given a Scale and min/max domain limits."""
    # This is an internal maplotlib function that simplifies things to access
    # It is likely to become part of the matplotlib API at some point:
    # https://github.com/matplotlib/matplotlib/issues/20329
    if isinstance(norm, mpl.colors.Normalize):
        return norm

    if norm is None:
        vmin = vmax = None
    else:
        vmin, vmax = norm  # TODO more helpful error if this fails?

    class ScaledNorm(mpl.colors.Normalize):

        def __call__(self, value, clip=None):
            # From github.com/matplotlib/matplotlib/blob/v3.4.2/lib/matplotlib/colors.py
            # See github.com/matplotlib/matplotlib/tree/v3.4.2/LICENSE
            value, is_scalar = self.process_value(value)
            self.autoscale_None(value)
            if self.vmin > self.vmax:
                raise ValueError("vmin must be less or equal to vmax")
            if self.vmin == self.vmax:
                return np.full_like(value, 0)
            if clip is None:
                clip = self.clip
            if clip:
                value = np.clip(value, self.vmin, self.vmax)
            # ***** Seaborn changes start ****
            t_value = self.transform(value).reshape(np.shape(value))
            t_vmin, t_vmax = self.transform([self.vmin, self.vmax])
            # ***** Seaborn changes end *****
            if not np.isfinite([t_vmin, t_vmax]).all():
                raise ValueError("Invalid vmin or vmax")
            t_value -= t_vmin
            t_value /= (t_vmax - t_vmin)
            t_value = np.ma.masked_invalid(t_value, copy=False)
            return t_value[0] if is_scalar else t_value

    new_norm = ScaledNorm(vmin, vmax)

    new_norm.transform = scale.get_transform().transform

    return new_norm


def scale_factory(scale, axis, **kwargs):
    """
    Backwards compatability for creation of independent scales.

    Matplotlib scales require an Axis object for instantiation on < 3.4.
    But the axis is not used, aside from extraction of the axis_name in LogScale.

    """
    if isinstance(scale, str):
        class Axis:
            axis_name = axis
        axis = Axis()
    return mpl.scale.scale_factory(scale, axis, **kwargs)


def set_scale_obj(ax, axis, scale):
    """Handle backwards compatability with setting matplotlib scale."""
    if LooseVersion(mpl.__version__) < "3.4":
        # The ability to pass a BaseScale instance to Axes.set_{}scale was added
        # to matplotlib in version 3.4.0: GH: matplotlib/matplotlib/pull/19089
        # Workaround: use the scale name, which is restrictive only if the user
        # wants to define a custom scale; they'll need to update the registry too.
        ax.set(**{f"{axis}scale": scale.scale_obj.name})
    else:
        ax.set(**{f"{axis}scale": scale.scale_obj})
