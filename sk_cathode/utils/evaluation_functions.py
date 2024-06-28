import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import colors
from sklearn.metrics import roc_curve


def sic_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Compute the significance improvement characteristic (SIC) curve.
    The function is based on sklearn.metrics.roc_curve and supports
    the same arguments.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    sic : array-like of shape (n_samples,)
        Increasing SIC values: True Positive Rate / sqrt(False Positive Rate),
        such that element i is the false positive rate of predictions with
        score >= thresholds[i].

    fpr : ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`.

    tpr : ndarray of shape (>2,)
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `np.inf`.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                         pos_label=pos_label,
                                         sample_weight=sample_weight)
        sic = tpr / np.sqrt(fpr)

    return sic, fpr, tpr, thresholds


def hists_2d(data, labels=None, weights=None, n_features=None, bins=30,
             ranges=None, color_marginal="blue", color_2d="Blues",
             out_file=None, suppress_show=False, title=None):
    """
    Plot 2D histograms and marginal distributions for a given dataset.

    Parameters
    ----------
    data: array-like of shape (n_samples, n_features)
        The data to be plotted.

    labels: list of str, default=None
        The labels for the features.

    weights: array-like of shape (n_samples,), default=None
        The weights for the data points.

    n_features: int, default=None
        The number of features to be plotted. Only needed if not all
        features in the data array should be plotted.

    bins: int or sequence of scalars, default=30
        The number of bins to use for the histograms.

    ranges: list of tuple, default=None
        The range for the features to be plotted.

    color_marginal: str, default="blue"
        The color for the marginal distributions.

    color_2d: str, default="Blues"
        The color map for the 2D correlation histograms.

    out_file: str, default=None
        The path to save the plot.

    suppress_show: bool, default=False
        Whether to suppress showing the plot interactively.

    title: str, default=None
        Title to draw into the figure.

    Returns
    -------
    None
    """

    if n_features is None:
        n_features = data.shape[1]

    if labels is None:
        labels = ["feature {}".format(i) for i in range(n_features)]

    if ranges is None:
        ranges = [None for _ in range(n_features)]
    else:
        assert len(ranges) == n_features

    fig = plt.figure(figsize=(n_features*2, n_features*2))
    gs = gridspec.GridSpec(n_features, n_features, wspace=0.03, hspace=0.03)

    if title is not None:
        plt.suptitle(title)

    binning = {}

    for i in range(n_features):
        plt.subplot(gs[i, i])
        _, binning[i], _ = plt.hist(data[:, i], bins=bins,
                                    range=ranges[i],
                                    weights=weights,
                                    color=color_marginal)
        plt.yscale("log")

        if i != n_features-1:
            plt.xticks([])
        else:
            plt.xlabel(labels[i])
        if i != 0:
            plt.yticks([])

    for i in range(n_features):

        for j in range(n_features):

            if j >= i:
                continue

            plt.subplot(gs[i, j])
            last_hist = plt.hist2d(
                data[:, j], data[:, i], bins=[binning[j], binning[i]],
                weights=weights, cmap=color_2d, norm=colors.LogNorm())[3]

            if i != n_features-1:
                plt.xticks([])
            else:
                plt.xlabel(labels[j])

            if j != 0:
                plt.yticks([])
            else:
                plt.ylabel(labels[i])

    cbar_ax = plt.subplot(gs[n_features-2, n_features-1])
    cbar = fig.colorbar(last_hist, cax=cbar_ax)
    cbar.set_label("counts", rotation=270)

    if out_file is not None:
        plt.savefig(out_file)
    if not suppress_show:
        plt.show()


def pulls_2d(data_ref, data_test, labels=None, bins=30,
             ranges=None, reference_binning=True,
             weights_ref=None, weights_test=None,
             n_features=None, out_file=None, suppress_show=False):
    """
    Plot pulls between two datasets, in order to spot difference between the
    distributions in 1D marginals and 2D correlations, i.e., for each bin it
    draws (N_ref-N_test)/uncertainty. where N_ref is directly the number of
    bin entries in data_ref and N_test arises from scaling data_test to the
    same integral first. The uncertainty is the statistical uncertainty of
    both N_ref and N_test. It is used to s

    Parameters
    ----------
    data_ref: array-like of shape (n_samples, n_features)
        The "reference data" to be plotted.

    data_test: array-like of shape (n_samples, n_features)
        The "test data" to be plotted.

    labels: list of str, default=None
        The labels for the features.

    bins: int or sequence of scalars, default=30
        The number of bins to use for the histograms.

    ranges: list of tuple, default=None
        The range for the features to be plotted.

    reference_binning: bool, default=True
        Whether to use the binning of the "reference data" for the "test data".

    weights_ref: array-like of shape (n_samples,), default=None
        The weights for the "reference data" points.

    weights_test: array-like of shape (n_samples,), default=None
        The weights for the "test data" points.

    n_features: int, default=None
        The number of features to be plotted. Only needed if not all
        features in the data array should be plotted.

    out_file: str, default=None
        The path to save the plot.

    suppress_show: bool, default=False
        Whether to suppress showing the plot interactively.

    Returns
    -------
    None
    """

    if n_features is None:
        assert data_ref.shape[1] == data_test.shape[1]
        n_features = data_ref.shape[1]

    if labels is None:
        labels = ["feature {}".format(i) for i in range(n_features)]

    if ranges is None:
        ranges = [None for _ in range(n_features)]
    else:
        assert len(ranges) == n_features

    fig = plt.figure(figsize=(n_features*2, n_features*2))
    gs = gridspec.GridSpec(n_features, n_features, wspace=0.03, hspace=0.03)

    binning = {}

    for i in range(n_features):

        plt.subplot(gs[i, i])

        if reference_binning:
            hist_ref, binning[i] = np.histogram(data_ref[:, i], bins=bins,
                                                range=ranges[i],
                                                weights=weights_ref)
            hist_test, _ = np.histogram(data_test[:, i], bins=binning[i],
                                        weights=weights_test)
        else:
            hist_test, binning[i] = np.histogram(data_test[:, i], bins=bins,
                                                 range=ranges[i],
                                                 weights=weights_test)
            hist_ref, _ = np.histogram(data_ref[:, i], bins=binning[i],
                                       weights=weights_ref)

        scale_factor_test = np.sum(hist_test) * np.diff(binning[i])
        scale_factor_ref = np.sum(hist_ref) * np.diff(binning[i])

        correction_factor = scale_factor_ref / scale_factor_test

        uncerts = np.sqrt(hist_ref + correction_factor**2 * hist_test)

        pulls = (hist_ref - hist_test * correction_factor) / uncerts

        plt.bar(0.5*(binning[i][:-1]+binning[i][1:])[pulls > 0],
                pulls[pulls > 0],
                color="red", width=0.95*np.diff(binning[i])[0])
        plt.bar(0.5*(binning[i][:-1]+binning[i][1:])[pulls < 0],
                pulls[pulls < 0],
                color="blue", width=0.95*np.diff(binning[i])[0])

        plt.axhline(1., linestyle=":", alpha=0.8,
                    color=plt.rcParams["text.color"])
        plt.axhline(0., linestyle="-", alpha=0.8,
                    color=plt.rcParams["text.color"])
        plt.axhline(-1., linestyle=":", alpha=0.8,
                    color=plt.rcParams["text.color"])
        plt.ylim(-3, 3)
        if ranges[i] is not None:
            plt.xlim(*ranges[i])

        if i != n_features-1:
            plt.xticks([])
        else:
            plt.xlabel(labels[i])
        if i != 0:
            plt.yticks([])

    for i in range(n_features):

        for j in range(n_features):

            if j >= i:
                continue

            plt.subplot(gs[i, j])

            hist_test, bins_x, bins_y = np.histogram2d(
                data_test[:, j], data_test[:, i],
                weights=weights_test,
                bins=[binning[j], binning[i]])
            hist_ref, _, _ = np.histogram2d(
                data_ref[:, j], data_ref[:, i],
                weights=weights_ref,
                bins=[binning[j], binning[i]])

            scale_factor_test = np.sum(hist_test) * np.outer(np.diff(bins_x),
                                                             np.diff(bins_y))
            scale_factor_ref = np.sum(hist_ref) * np.outer(np.diff(bins_x),
                                                           np.diff(bins_y))

            correction_factor = scale_factor_ref / scale_factor_test

            uncerts = np.sqrt(hist_ref + correction_factor**2 * hist_test)

            pulls = (hist_ref - hist_test * correction_factor) / uncerts

            im = plt.imshow(np.transpose(pulls), cmap="bwr",
                            norm=colors.Normalize(vmin=-3, vmax=3),
                            origin="lower", extent=[bins_x[0], bins_x[-1],
                                                    bins_y[0], bins_y[-1]],
                            aspect="auto")

            if i != n_features-1:
                plt.xticks([])
            else:
                plt.xlabel(labels[j])

            if j != 0:
                plt.yticks([])
            else:
                plt.ylabel(labels[i])

    if weights_test is not None or weights_ref is not None:
        weights_ax = plt.subplot(gs[n_features-4, n_features-1])
        if weights_test is not None:
            weights_ax.hist(weights_test, bins=100, range=(0., 2.),
                            histtype="step", color="cornflowerblue",
                            label="test")
        if weights_ref is not None:
            weights_ax.hist(weights_ref, bins=100, range=(0., 2.),
                            histtype="step", color="orangered",
                            label="reference")
        weights_ax.set_xlabel("weights")
        weights_ax.set_yscale("log")

    cbar_ax = plt.subplot(gs[n_features-2, n_features-1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("pulls", rotation=270)

    if out_file is not None:
        plt.savefig(out_file)
    if not suppress_show:
        plt.show()


def preds_2d(data, data_preds, labels=None, bins=30, ranges=None,
             centering=0.5, preds_range=(-0.05, 0.05),
             n_features=None, out_file=None, suppress_show=False):
    """
    Plot the predictions of a model as a function of the features.
    Shows the features in 1D, as well as the 2D correlations between them.

    Parameters
    ----------
    data: array-like of shape (n_samples, n_features)
        The data containing the features to be plotted.

    data_preds: array-like of shape (n_samples,)
        The model predictions for the data.

    labels: list of str, default=None
        The labels for the features.

    bins: int or sequence of scalars, default=30
        The number of bins to use for the histograms.

    ranges: list of tuple, default=None
        The range for the features to be plotted.

    centering: float, 'median' or 'mean', default=0.5
        The value to subtract from the predictions.

    preds_range: tuple, default=(-0.05, 0.05)
        The range for the predictions (after subtraction).

    n_features: int, default=None
        The number of features to be plotted. Only needed if not all
        features in the data array should be plotted.

    out_file: str, default=None
        The path to save the plot.

    suppress_show: bool, default=False
        Whether to suppress showing the plot interactively.

    Returns
    -------
    None
    """

    if n_features is None:
        n_features = data.shape[1]

    if labels is None:
        labels = ["feature {}".format(i) for i in range(n_features)]

    if ranges is None:
        ranges = [None for _ in range(n_features)]
    else:
        assert len(ranges) == n_features

    if isinstance(centering, float):
        shift = centering
    elif centering == "median":
        shift = np.median(data_preds)
    elif centering == "mean":
        shift = 0.5
    else:
        raise ValueError("centering must be float, 'median' or 'mean'")

    fig = plt.figure(figsize=(n_features*2, n_features*2))
    gs = gridspec.GridSpec(n_features, n_features, wspace=0.03, hspace=0.03)

    binning = {}

    for i in range(n_features):

        plt.subplot(gs[i, i])
        _, binning[i] = np.histogram(data[:, i], bins=bins,
                                     range=ranges[i])

        profile, _ = profile_1d(data[:, i], data_preds, binning[i])
        norm_profile = profile - shift

        plt.bar(0.5*(binning[i][:-1]+binning[i][1:])[norm_profile > 0],
                norm_profile[norm_profile > 0],
                color="red", width=0.95*np.diff(binning[i])[0])
        plt.bar(0.5*(binning[i][:-1]+binning[i][1:])[norm_profile < 0],
                norm_profile[norm_profile < 0],
                color="blue", width=0.95*np.diff(binning[i])[0])
        plt.axhline(0., linestyle="-", alpha=0.8,
                    color=plt.rcParams["text.color"])
        plt.ylim(*preds_range)
        if ranges[i] is not None:
            plt.xlim(*ranges[i])

        if i != n_features-1:
            plt.xticks([])
        else:
            plt.xlabel(labels[i])
        if i != 0:
            plt.yticks([])
        else:
            plt.ylabel("preds\N{MINUS SIGN}"+str(centering))

    for i in range(n_features):

        for j in range(n_features):

            if j >= i:
                continue

            plt.subplot(gs[i, j])

            profile, _, _ = profile_2d(
                data[:, j], data[:, i], data_preds, [binning[j], binning[i]]
            )
            norm_profile = profile - shift

            im = plt.imshow(np.transpose(norm_profile),
                            cmap="bwr", origin="lower",
                            vmin=preds_range[0], vmax=preds_range[1],
                            extent=[binning[j][0], binning[j][-1],
                                    binning[i][0], binning[i][-1]],
                            aspect="auto")

            if i != n_features-1:
                plt.xticks([])
            else:
                plt.xlabel(labels[j])

            if j != 0:
                plt.yticks([])
            else:
                plt.ylabel(labels[i])

    preds_ax = plt.subplot(gs[n_features-4, n_features-1])
    colormap_hist(data_preds, plt.get_cmap("bwr"), bins, ax=preds_ax,
                  color_range=[x + shift for x in preds_range])
    preds_ax.set_xlabel("preds")
    preds_ax.set_yscale("log")

    cbar_ax = plt.subplot(gs[n_features-2, n_features-1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("preds\N{MINUS SIGN}"+str(centering), rotation=270)

    if out_file is not None:
        plt.savefig(out_file)
    if not suppress_show:
        plt.show()


def profile_1d(data, preds, bins):
    """
    Helper function to compute the average predictions for the data
    in each bin, in 1D.
    """
    hist, edges = np.histogram(data, bins=bins)
    avg_preds = np.zeros_like(hist, dtype=float)
    avg_preds[:] = np.nan

    for i in range(len(hist)):
        mask = (data >= edges[i]) & (data < edges[i+1])
        if np.any(mask):
            avg_preds[i] = np.mean(preds[mask])

    return avg_preds, edges


def profile_2d(data_x, data_y, preds, bins):
    """
    Helper function to compute the average predictions for the data
    in each bin, in 2D.
    """

    hist, xedges, yedges = np.histogram2d(data_x, data_y, bins=bins)
    avg_preds = np.zeros_like(hist)
    avg_preds[:] = np.nan

    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            mask = (data_x >= xedges[i]) & (data_x < xedges[i+1]) & \
                   (data_y >= yedges[j]) & (data_y < yedges[j+1])
            if np.any(mask):
                avg_preds[i, j] = np.mean(preds[mask])

    return avg_preds, xedges, yedges


def rescale(x, range=(0.45, 0.55)):
    """
    Helper function to rescale the input to the range [0, 1].
    """
    return (x - range[0]) / (range[1] - range[0])


def colormap_hist(data, colormap, bins, ax=None,
                  color_range=(0, 1), plot_range=None):
    """
    Helper function to plot a 1D histogram with a given colormap.
    """

    if ax is None:
        ax = plt.gca()

    hist, binning = np.histogram(data, bins=bins, range=plot_range)
    x_vals = 0.5*(binning[:-1]+binning[1:])
    color = colormap(rescale(x_vals, range=color_range))
    return ax.bar(x_vals, hist, color=color,
                  width=0.95*np.diff(binning)[0])
