"""Algorithms to support fitting routines in seaborn plotting functions."""
from __future__ import division
import numpy as np
from scipy import stats
from .external.six.moves import range


def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and store aggregate values.

    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.

    Keyword arguments:
        n_boot : int, default 10000
            Number of iterations
        axis : int, default None
            Will pass axis to ``func`` as a keyword argument.
        units : array, default None
            Array of sampling unit IDs. When used the bootstrap resamples units
            and then observations within units instead of individual
            datapoints.
        smooth : bool, default False
            If True, performs a smoothed bootstrap (draws samples from a kernel
            destiny estimate); only works for one-dimensional inputs and cannot
            be used `units` is present.
        func : callable, default np.mean
            Function to call on the args that are passed in.
        random_seed : int | None, default None
            Seed for the random number generator; useful if you want
            reproducible resamples.

    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values

    """
    # Ensure list of arrays are same length
    if len(np.unique(list(map(len, args)))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", np.mean)
    axis = kwargs.get("axis", None)
    units = kwargs.get("units", None)
    smooth = kwargs.get("smooth", False)
    random_seed = kwargs.get("random_seed", None)
    if axis is None:
        func_kwargs = dict()
    else:
        func_kwargs = dict(axis=axis)

    # Initialize the resampler
    rs = np.random.RandomState(random_seed)

    # Coerce to arrays
    args = list(map(np.asarray, args))
    if units is not None:
        units = np.asarray(units)

    # Do the bootstrap
    if smooth:
        return _smooth_bootstrap(args, n_boot, func, func_kwargs)

    if units is not None:
        return _structured_bootstrap(args, n_boot, units, func,
                                     func_kwargs, rs)

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = rs.randint(0, n, n)
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def percentiles(a, pcts, axis=None):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a : array
        data
    pcts : sequence of percentile values
        percentile or percentiles to find score at
    axis : int or None
        if not None, computes scores over this axis

    Returns
    -------
    scores : array
        array of scores at requested percentiles
        first dimension is length of object passed to ``pcts``

    """
    scores = []
    try:
        n = len(pcts)
    except TypeError:
        pcts = [pcts]
        n = 0
    for i, p in enumerate(pcts):
        if axis is None:
            score = stats.scoreatpercentile(a.ravel(), p)
        else:
            score = np.apply_along_axis(stats.scoreatpercentile, axis, a, p)
        scores.append(score)
    scores = np.asarray(scores)
    if not n:
        scores = scores.squeeze()
    return scores


def acceleration(data):
    '''
    Compute the acceleration statistic

    Parameters
    ----------
        data : 1-D numpy array

    Returns
    -------
        acc (float) : the acceleration statistic

    '''
    # intermediate values
    SSD = np.sum((data.mean() - data)**3)
    SCD = np.sum((data.mean() - data)**2)

    # dodge the ZeroDivision error
    if SCD == 0:
        SCD = 1e-12

    # comput and return the acceleration
    return SSD / (6 * SCD**1.5)


def ci(a, which=95, axis=None, how='percentile', refval=None):
    """Return a confidence interval from an array of (bootstrapped)
    values.

    Parameters
    ----------
    a : array-like
        A sequence of bootstrapped statistic for which the confidence
        interval will be computed.

    which : optional float (default = 95)
        The level of confidence of the intervals.

    axis : optional int or None (default)
        If not None, computes scores over this axis when using the
        percentile method.

    how : optional string ("percentile" or "bca")
        Selects the method for computing the confidence interval.
        BCa = "bias-corrected and accelerated". See References section.

    refval : optional float or None (default)
        Baseline result for the BCa method. Typically this the result of
        the statistical function fed into `bootstrap` applied to the
        original data.

    Returns
    -------
    CI : sequence of lower and upper confidence bounds.

    References
    ----------
    DiCiccio, T.J. and Efron. B. (1996). "Bootstrap Confidence
        Intervals." Statistical Sciences, Vol 11, No.3, pp189-228.
    As of March 2014, available at:
    http://staff.ustc.edu.cn/~zwp/teach/Stat-Comp/Efron_Bootstrap_CIs.pdf

    """
    p = np.array([50 - which / 2, 50 + which / 2])
    if how.lower() == 'percentile':
        CI = percentiles(a, p, axis)

    elif how.lower() == 'bca':
        if refval is None:
            refval = np.mean(a)

        n_below = np.sum(a < refval)
        if n_below == 0:
            n_below = 0.00001

        # z-stats on the % of `n_below` and the confidence limits
        z0 = stats.distributions.norm.ppf(float(n_below)/len(a))
        z = stats.distributions.norm.ppf(p / 100.0)

        # compute the acceleration
        a_hat = acceleration(a)

        # refine the confidence limits (alphas)
        zTotal = (z0 + (z0 + z)) / (1 - a_hat*(z0+z))
        alpha = stats.distributions.norm.cdf(zTotal) * 100.0

        # confidence intervals from the new alphas
        CI = percentiles(a, alpha)

        # fall back to the standard percentile method if the CIs
        # don't make any sense (i.e., don't surround refval)
        if refval < CI[0] or CI[1] < refval or n_below == len(a):
            CI = ci(a, which=which, axis=axis, how='percentile')

    else:
        raise ValueError("`how` must be either 'BCa' or 'percentile'")

    return CI


def _structured_bootstrap(args, n_boot, units, func, func_kwargs, rs):
    """Resample units instead of datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)

    args = [[a[units == unit] for unit in unique_units] for a in args]

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = rs.randint(0, n_units, n_units)
        sample = [np.take(a, resampler, axis=0) for a in args]
        lengths = map(len, sample[0])
        resampler = [rs.randint(0, n, n) for n in lengths]
        sample = [[c.take(r, axis=0) for c, r in zip(a, resampler)]
                  for a in sample]
        sample = list(map(np.concatenate, sample))
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def _smooth_bootstrap(args, n_boot, func, func_kwargs):
    """Bootstrap by resampling from a kernel density estimate."""
    n = len(args[0])
    boot_dist = []
    kde = [stats.gaussian_kde(np.transpose(a)) for a in args]
    for i in range(int(n_boot)):
        sample = [a.resample(n).T for a in kde]
        boot_dist.append(func(*sample, **func_kwargs))
    return np.array(boot_dist)


def randomize_corrmat(a, tail="both", corrected=True, n_iter=1000,
                      random_seed=None, return_dist=False):
    """Test the significance of set of correlations with permutations.

    By default this corrects for multiple comparisons across one side
    of the matrix.

    Parameters
    ----------
    a : n_vars x n_obs array
        array with variables as rows
    tail : both | upper | lower
        whether test should be two-tailed, or which tail to integrate over
    corrected : boolean
        if True reports p values with respect to the max stat distribution
    n_iter : int
        number of permutation iterations
    random_seed : int or None
        seed for RNG
    return_dist : bool
        if True, return n_vars x n_vars x n_iter

    Returns
    -------
    p_mat : float
        array of probabilites for actual correlation from null CDF

    """
    if tail not in ["upper", "lower", "both"]:
        raise ValueError("'tail' must be 'upper', 'lower', or 'both'")

    rs = np.random.RandomState(random_seed)

    a = np.asarray(a, np.float)
    flat_a = a.ravel()
    n_vars, n_obs = a.shape

    # Do the permutations to establish a null distribution
    null_dist = np.empty((n_vars, n_vars, n_iter))
    for i_i in range(n_iter):
        perm_i = np.concatenate([rs.permutation(n_obs) + (v * n_obs)
                                 for v in range(n_vars)])
        a_i = flat_a[perm_i].reshape(n_vars, n_obs)
        null_dist[..., i_i] = np.corrcoef(a_i)

    # Get the observed correlation values
    real_corr = np.corrcoef(a)

    # Figure out p values based on the permutation distribution
    p_mat = np.zeros((n_vars, n_vars))
    upper_tri = np.triu_indices(n_vars, 1)

    if corrected:
        if tail == "both":
            max_dist = np.abs(null_dist[upper_tri]).max(axis=0)
        elif tail == "lower":
            max_dist = null_dist[upper_tri].min(axis=0)
        elif tail == "upper":
            max_dist = null_dist[upper_tri].max(axis=0)

        cdf = lambda x: stats.percentileofscore(max_dist, x) / 100.

        for i, j in zip(*upper_tri):
            observed = real_corr[i, j]
            if tail == "both":
                p_ij = 1 - cdf(abs(observed))
            elif tail == "lower":
                p_ij = cdf(observed)
            elif tail == "upper":
                p_ij = 1 - cdf(observed)
            p_mat[i, j] = p_ij

    else:
        for i, j in zip(*upper_tri):

            null_corrs = null_dist[i, j]
            cdf = lambda x: stats.percentileofscore(null_corrs, x) / 100.

            observed = real_corr[i, j]
            if tail == "both":
                p_ij = 2 * (1 - cdf(abs(observed)))
            elif tail == "lower":
                p_ij = cdf(observed)
            elif tail == "upper":
                p_ij = 1 - cdf(observed)
            p_mat[i, j] = p_ij

    # Make p matrix symettrical with nans on the diagonal
    p_mat += p_mat.T
    p_mat[np.diag_indices(n_vars)] = np.nan

    if return_dist:
        return p_mat, null_dist
    return p_mat
