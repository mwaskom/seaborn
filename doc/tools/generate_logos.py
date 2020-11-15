import numpy as np
import seaborn as sns
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.spatial import distance


XY_CACHE = {}

STATIC_DIR = "_static"
plt.rcParams["savefig.dpi"] = 300


def poisson_disc_sample(array_radius, pad_radius, candidates=100, d=2, seed=None):
    """Find positions using poisson-disc sampling."""
    # See http://bost.ocks.org/mike/algorithms/
    rng = np.random.default_rng(seed)
    uniform = rng.uniform
    randint = rng.integers

    # Cache the results
    key = array_radius, pad_radius, seed
    if key in XY_CACHE:
        return XY_CACHE[key]

    # Start at a fixed point we know will work
    start = np.zeros(d)
    samples = [start]
    queue = [start]

    while queue:

        # Pick a sample to expand from
        s_idx = randint(len(queue))
        s = queue[s_idx]

        for i in range(candidates):
            # Generate a candidate from this sample
            coords = uniform(s - 2 * pad_radius, s + 2 * pad_radius, d)

            # Check the three conditions to accept the candidate
            in_array = np.sqrt(np.sum(coords ** 2)) < array_radius
            in_ring = np.all(distance.cdist(samples, [coords]) > pad_radius)

            if in_array and in_ring:
                # Accept the candidate
                samples.append(coords)
                queue.append(coords)
                break

        if (i + 1) == candidates:
            # We've exhausted the particular sample
            queue.pop(s_idx)

    samples = np.array(samples)
    XY_CACHE[key] = samples
    return samples


def logo(
    ax,
    color_kws, ring, ring_idx, edge,
    pdf_means, pdf_sigma, dy, y0, w, h,
    hist_mean, hist_sigma, hist_y0, lw, skip,
    scatter, pad, scale,
):

    # Square, invisible axes with specified limits to center the logo
    ax.set(xlim=(35 + w, 95 - w), ylim=(-3, 53))
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Magic numbers for the logo circle
    radius = 27
    center = 65, 25

    # Full x and y grids for a gaussian curve
    x = np.arange(101)
    y = gaussian(x.size, pdf_sigma)

    x0 = 30  # Magic number
    xx = x[x0:]

    # Vertical distances between the PDF curves
    n = len(pdf_means)
    dys = np.linspace(0, (n - 1) * dy, n) - (n * dy / 2)
    dys -= dys.mean()

    # Compute the PDF curves with vertical offsets
    pdfs = [h * (y[x0 - m:-m] + y0 + dy) for m, dy in zip(pdf_means, dys)]

    # Add in constants to fill from bottom and to top
    pdfs.insert(0, np.full(xx.shape, -h))
    pdfs.append(np.full(xx.shape, 50 + h))

    # Color gradient
    colors = sns.cubehelix_palette(n + 1 + bool(hist_mean), **color_kws)

    # White fill between curves and around edges
    bg = patches.Circle(
        center, radius=radius - 1 + ring, color="white",
        transform=ax.transData, zorder=0,
    )
    ax.add_artist(bg)

    # Clipping artist (not shown) for the interior elements
    fg = patches.Circle(center, radius=radius - edge, transform=ax.transData)

    # Ring artist to surround the circle (optional)
    if ring:
        wedge = patches.Wedge(
            center, r=radius + edge / 2, theta1=0, theta2=360, width=edge / 2,
            transform=ax.transData, color=colors[ring_idx], alpha=1
        )
        ax.add_artist(wedge)

    # Add histogram bars
    if hist_mean:
        hist_color = colors.pop(0)
        hist_y = gaussian(x.size, hist_sigma)
        hist = 1.1 * h * (hist_y[x0 - hist_mean:-hist_mean] + hist_y0)
        dx = x[skip] - x[0]
        hist_x = xx[::skip]
        hist_h = h + hist[::skip]
        # Magic number to avoid tiny sliver of bar on edge
        use = hist_x < center[0] + radius * .5
        bars = ax.bar(
            hist_x[use], hist_h[use], bottom=-h, width=dx,
            align="edge", color=hist_color, ec="w", lw=lw,
            zorder=3,
        )
        for bar in bars:
            bar.set_clip_path(fg)

    # Add each smooth PDF "wave"
    for i, pdf in enumerate(pdfs[1:], 1):
        u = ax.fill_between(xx, pdfs[i - 1] + w, pdf, color=colors[i - 1], lw=0)
        u.set_clip_path(fg)

    # Add scatterplot in top wave area
    if scatter:
        seed = sum(map(ord, "seaborn logo"))
        xy = poisson_disc_sample(radius - edge - ring, pad, seed=seed)
        clearance = distance.cdist(xy + center, np.c_[xx, pdfs[-2]])
        use = clearance.min(axis=1) > pad / 1.8
        x, y = xy[use].T
        sizes = (x - y) % 9

        points = ax.scatter(
            x + center[0], y + center[1], s=scale * (10 + sizes * 5),
            zorder=5, color=colors[-1], ec="w", lw=scale / 2,
        )
        path = u.get_paths()[0]
        points.set_clip_path(path, transform=u.get_transform())
        u.set_visible(False)


def savefig(fig, shape, variant):

    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    facecolor = (1, 1, 1, 1) if bg == "white" else (1, 1, 1, 0)

    for ext in ["png", "svg"]:
        fig.savefig(f"{STATIC_DIR}/logo-{shape}-{variant}bg.{ext}", facecolor=facecolor)


if __name__ == "__main__":

    for bg in ["white", "light", "dark"]:

        color_idx = -1 if bg == "dark" else 0

        kwargs = dict(
            color_kws=dict(start=.3, rot=-.4, light=.8, dark=.3, reverse=True),
            ring=True, ring_idx=color_idx, edge=1,
            pdf_means=[8, 24], pdf_sigma=16,
            dy=1, y0=1.8, w=.5, h=12,
            hist_mean=2, hist_sigma=10, hist_y0=.6, lw=1, skip=6,
            scatter=True, pad=1.8, scale=.5,
        )
        color = sns.cubehelix_palette(**kwargs["color_kws"])[color_idx]

        # ------------------------------------------------------------------------ #

        fig, ax = plt.subplots(figsize=(2, 2), facecolor="w", dpi=100)
        logo(ax, **kwargs)
        savefig(fig, "mark", bg)

        # ------------------------------------------------------------------------ #

        fig, axs = plt.subplots(1, 2, figsize=(8, 2), dpi=100,
                                gridspec_kw=dict(width_ratios=[1, 3]))
        logo(axs[0], **kwargs)

        font = {
            "family": "avenir",
            "color": color,
            "weight": "regular",
            "size": 120,
        }
        axs[1].text(.01, .35, "seaborn", ha="left", va="center",
                    fontdict=font, transform=axs[1].transAxes)
        axs[1].set_axis_off()
        savefig(fig, "wide", bg)

        # ------------------------------------------------------------------------ #

        fig, axs = plt.subplots(2, 1, figsize=(2, 2.5), dpi=100,
                                gridspec_kw=dict(height_ratios=[4, 1]))

        logo(axs[0], **kwargs)

        font = {
            "family": "avenir",
            "color": color,
            "weight": "regular",
            "size": 34,
        }
        axs[1].text(.5, 1, "seaborn", ha="center", va="top",
                    fontdict=font, transform=axs[1].transAxes)
        axs[1].set_axis_off()
        savefig(fig, "tall", bg)
