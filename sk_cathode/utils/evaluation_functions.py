import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from matplotlib import colors


def hists_2d(data, labels=None, weights=None, n_features=None, bins=30,
             ranges=None, color_marginal="blue", color_2d="Blues",
             out_file=None, suppress_show=False, title=None):

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


def pulls_2d(data_num, data_denom, labels=None, bins=30,
             ranges=None, numerator_binning=True,
             weights_num=None, weights_denom=None,
             n_features=None, out_file=None, suppress_show=False):

    if n_features is None:
        assert data_num.shape[1] == data_denom.shape[1]
        n_features = data_num.shape[1]

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

        if numerator_binning:
            hist_num, binning[i] = np.histogram(data_num[:, i], bins=bins,
                                                range=ranges[i],
                                                weights=weights_num)
            hist_denom, _ = np.histogram(data_denom[:, i], bins=binning[i],
                                         weights=weights_denom)
        else:
            hist_denom, binning[i] = np.histogram(data_denom[:, i], bins=bins,
                                                  range=ranges[i],
                                                  weights=weights_denom)
            hist_num, _ = np.histogram(data_num[:, i], bins=binning[i],
                                       weights=weights_num)

        scale_factor_denom = np.sum(hist_denom) * np.diff(binning[i])
        scale_factor_num = np.sum(hist_num) * np.diff(binning[i])

        correction_factor = scale_factor_num / scale_factor_denom

        uncerts = np.sqrt(hist_num + correction_factor**2 * hist_denom)

        pulls = (hist_num - hist_denom * correction_factor) / uncerts

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

            hist_denom, bins_x, bins_y = np.histogram2d(
                data_denom[:, j], data_denom[:, i],
                weights=weights_denom,
                bins=[binning[j], binning[i]])
            hist_num, _, _ = np.histogram2d(
                data_num[:, j], data_num[:, i],
                weights=weights_num,
                bins=[binning[j], binning[i]])

            scale_factor_denom = np.sum(hist_denom) * np.outer(np.diff(bins_x),
                                                               np.diff(bins_y))
            scale_factor_num = np.sum(hist_num) * np.outer(np.diff(bins_x),
                                                           np.diff(bins_y))

            correction_factor = scale_factor_num / scale_factor_denom

            uncerts = np.sqrt(hist_num + correction_factor**2 * hist_denom)

            pulls = (hist_num - hist_denom * correction_factor) / uncerts

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

    if weights_denom is not None or weights_num is not None:
        weights_ax = plt.subplot(gs[n_features-4, n_features-1])
        if weights_denom is not None:
            weights_ax.hist(weights_denom, bins=100, range=(0., 2.),
                            histtype="step", color="cornflowerblue",
                            label="denom")
        if weights_num is not None:
            weights_ax.hist(weights_num, bins=100, range=(0., 2.),
                            histtype="step", color="orangered",
                            label="num")
        weights_ax.set_xlabel("weights")
        weights_ax.set_yscale("log")

    cbar_ax = plt.subplot(gs[n_features-2, n_features-1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("pulls", rotation=270)

    if out_file is not None:
        plt.savefig(out_file)
    if not suppress_show:
        plt.show()


def profile_1d(data, preds, bins):
    hist, edges = np.histogram(data, bins=bins)
    avg_preds = np.zeros_like(hist, dtype=float)
    avg_preds[:] = np.nan

    for i in range(len(hist)):
        mask = (data >= edges[i]) & (data < edges[i+1])
        if np.any(mask):
            avg_preds[i] = np.mean(preds[mask])

    return avg_preds, edges


def profile_2d(data_x, data_y, preds, bins):

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
    return (x - range[0]) / (range[1] - range[0])


def colormap_hist(data, colormap, bins, ax=None,
                  color_range=(0, 1), plot_range=None):

    if ax is None:
        ax = plt.gca()

    hist, binning = np.histogram(data, bins=bins, range=plot_range)
    x_vals = 0.5*(binning[:-1]+binning[1:])
    color = colormap(rescale(x_vals, range=color_range))
    return ax.bar(x_vals, hist, color=color,
                  width=0.95*np.diff(binning)[0])


def preds_2d(data, data_preds, labels=None, bins=30, ranges=None,
             centering=0.5, preds_range=(-0.05, 0.05),
             n_features=None, out_file=None, suppress_show=False):

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
