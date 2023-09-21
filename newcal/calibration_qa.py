import numpy as np
import sys
import time
import pyuvdata
from newcal import cost_function_calculations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing


def calculate_per_antenna_cost(
    gains,
    Nants,
    Nbls,
    Nfreqs,
    N_feed_pols,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    parallel=True,
):
    """
    Calculate the contribution of each antenna to the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Nfreqs : int
        Number of frequency channels.
    N_feed_pols : int
        Number of feed polarization modes to be fit.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,). Polarizations are ordered in
        the AIPS convention: XX, YY, XY, YX.
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,). Polarizations are ordered in
        the AIPS convention: XX, YY, XY, YX.
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    parallel : bool
        Set to True to parallelize with multiprocessing. Default True.

    Returns
    -------
    per_ant_cost_normalized : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    """

    per_ant_cost = np.zeros((Nants, N_feed_pols), dtype=float)
    per_ant_baselines = np.zeros((Nants, N_feed_pols), dtype=int)
    if parallel:
        args_list = []
    for pol_ind in range(N_feed_pols):
        total_visibilities = np.count_nonzero(visibility_weights[:, :, :, pol_ind])
        total_cost = 0.0
        for freq_ind in range(Nfreqs):
            if parallel:
                args = (
                    gains[:, freq_ind, pol_ind],
                    model_visibilities[:, :, freq_ind, pol_ind],
                    data_visibilities[:, :, freq_ind, pol_ind],
                    visibility_weights[:, :, freq_ind, pol_ind],
                    gains_exp_mat_1,
                    gains_exp_mat_2,
                    0.0,
                )
                args_list.append(args)
            else:
                total_cost += cost_function_calculations.cost_function_single_pol(
                    gains[:, freq_ind, pol_ind],
                    model_visibilities[:, :, freq_ind, pol_ind],
                    data_visibilities[:, :, freq_ind, pol_ind],
                    visibility_weights[:, :, freq_ind, pol_ind],
                    gains_exp_mat_1,
                    gains_exp_mat_2,
                    0.0,
                )
        for ant_ind in range(Nants):
            bl_inds = np.logical_or(
                gains_exp_mat_1[:, ant_ind],
                gains_exp_mat_2[:, ant_ind],
            )
            ant_excluded_weights = np.copy(visibility_weights[:, :, :, pol_ind])
            ant_excluded_weights[:, bl_inds, :] = 0
            per_ant_baselines[ant_ind, pol_ind] = total_visibilities - np.count_nonzero(
                ant_excluded_weights
            )
            if parallel:
                args = (
                    gains[:, freq_ind, pol_ind],
                    model_visibilities[:, :, freq_ind, pol_ind],
                    data_visibilities[:, :, freq_ind, pol_ind],
                    ant_excluded_weights[:, :, freq_ind],
                    gains_exp_mat_1,
                    gains_exp_mat_2,
                    0.0,
                )
                args_list.append(args)
            else:
                per_ant_cost[ant_ind, pol_ind] = total_cost
                for freq_ind in range(Nfreqs):
                    per_ant_cost[
                        ant_ind, pol_ind
                    ] -= cost_function_calculations.cost_function_single_pol(
                        gains[:, freq_ind, pol_ind],
                        model_visibilities[:, :, freq_ind, pol_ind],
                        data_visibilities[:, :, freq_ind, pol_ind],
                        ant_excluded_weights[:, :, freq_ind],
                        gains_exp_mat_1,
                        gains_exp_mat_2,
                        0.0,
                    )

    if parallel:
        pool = multiprocessing.Pool()
        result = pool.starmap(
            calibration_optimization.run_calibration_optimization_per_pol_single_freq,
            args_list,
        )
        pool.close()
        pool.join()
        result = result.reshape((N_feed_pols, Nants+1, Nfreqs))
        total_cost = np.sum(result[:, 0, :], axis=1)
        per_ant_cost = np.transpose(total_cost[:, np.newaxis] - np.sum(result[:, 1:, :], axis=2))

    per_ant_cost_normalized = np.abs(per_ant_cost / per_ant_baselines)
    return per_ant_cost_normalized


def plot_per_ant_cost(per_ant_cost, antenna_names, plot_output_dir, plot_prefix=""):
    """
    Plot the per-antenna cost.

    Parameters
    ----------
    per_ant_cost : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the per_ant_cost.
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    plot_prefix : str
        Optional string to be appended to the start of the file names.
    """

    # Format antenna names
    sort_inds = np.argsort(antenna_names)
    ant_names_sorted = antenna_names[sort_inds]
    per_ant_cost_sorted = per_ant_cost[sort_inds, :]
    ant_nums = np.array([int(name[3:]) for name in ant_names_sorted])

    # Parse strings
    use_plot_prefix = plot_prefix
    if len(plot_prefix) > 0:
        if not use_plot_prefix.endswith("_"):
            use_plot_prefix = f"{use_plot_prefix}_"
    use_plot_output_dir = plot_output_dir
    if plot_output_dir.endswith("/"):
        use_plot_output_dir = use_plot_output_dir[:-1]

    # Plot style parameters
    colors = ["tab:blue", "tab:orange"]
    linewidth = 0
    markersize = 0.5
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors[0],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="X",
        ),
        Line2D(
            [0],
            [0],
            color=colors[1],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="Y",
        ),
    ]

    fig, ax = plt.subplots()
    for pol_ind in range(np.shape(per_ant_cost_sorted)[1]):
        ax.plot(
            ant_nums,
            per_ant_cost_sorted[:, pol_ind],
            "-o",
            linewidth=linewidth,
            markersize=markersize,
            label=(["X", "Y"])[pol_ind],
            color=colors[pol_ind],
        )
    ax.set_ylim([0, np.nanmax(per_ant_cost_sorted[np.isfinite(per_ant_cost_sorted)])])
    ax.set_xlim([0, np.max(ant_nums)])
    ax.set_xlabel("Antenna Name")
    ax.set_ylabel("Per Antenna Cost Contribution")
    plt.legend(
        handles=legend_elements,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(
        f"{use_plot_output_dir}/{use_plot_prefix}per_ant_cost.png",
        dpi=600,
    )
    plt.close()


def get_antenna_flags_from_per_ant_cost(
    per_ant_cost,
    antenna_names,
    flagging_threshold=2.5,
    visibility_weights=None,
    gains_exp_mat_1=None,
    gains_exp_mat_2=None,
    feed_pols=["X", "Y"],
    vis_pols=["XX", "YY", "XY", "YX"],
):
    """
    Create a list of antennas to flag based on the per-antenna cost function.
    Option to update visibility_weights according to the flags.

    Parameters
    ----------
    per_ant_cost : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the per_ant_cost.
    flagging_threshold : float
        Flagging threshold. Per antenna cost values equal to flagging_threshold
        times the mean value will be flagged. Default 2.5.
    visibility_weights : None or array of float
        If not None, shape (Ntimes, Nbls, Nfreqs, N_vis_pols,). If
        visibility_weights, gains_exp_mat_1, and gains_exp_mat_2 are all not
        None, visibility_weights will be updated with the antenna flags. Default
        None.
    gains_exp_mat_1 : None or array of int
        If not None, shape (Nbls, Nants,). If visibility_weights,
        gains_exp_mat_1, and gains_exp_mat_2 are all not None,
        visibility_weights will be updated with the antenna flags. Default None.
    gains_exp_mat_2 : None or array of int
        If not None, shape (Nbls, Nants,). If visibility_weights,
        gains_exp_mat_1, and gains_exp_mat_2 are all not None,
        visibility_weights will be updated with the antenna flags. Default None.
    feed_pols : array of str
        Shape (N_feed_pols). Default ["X", "Y"]. Used only if visibility_weights
        is not None.
    vis_pols : array of str
        Shape (N_vis_pols). Default ["XX", "YY", "XY", "YX"]. Used only if
        visibility_weights is not None.

    Returns
    -------
    flag_antenna_list : list of arrays of str
        Length N_feed_pols. flag_antenna_list[pol_ind] provides an array of
        antenna names identified for flagging.
    visibility_weights : None or array of float
        If not None, shape (Ntimes, Nbls, Nfreqs, N_vis_pols,). If not None,
        updated visibility_weights that reflect the flagged antennas.
    """

    N_feed_pols = np.shape(per_ant_cost)[1]
    where_finite = np.isfinite(per_ant_cost)
    if np.sum(where_finite) > 0:
        mean_per_ant_cost = np.mean(per_ant_cost[where_finite])
        flag_antenna_list = []
        for pol_ind in range(N_feed_pols):
            flag_antenna_inds = np.where(
                np.logical_or(
                    per_ant_cost[:, pol_ind] > flagging_threshold * mean_per_ant_cost,
                    ~np.isfinite(per_ant_cost[:, pol_ind]),
                )
            )[0]
            flag_antenna_list.append(antenna_names[flag_antenna_inds])
            if (
                visibility_weights is not None
                and gains_exp_mat_1 is not None
                and gains_exp_mat_2 is not None
            ):
                use_vis_pol_inds = np.where(
                    np.array([pol_name[pol_ind] for pol_name in vis_pols])
                    == feed_pols[pol_ind]
                )[0]
                for ant_ind in flag_antenna_inds:
                    bl_inds = np.logical_or(
                        gains_exp_mat_1[:, ant_ind],
                        gains_exp_mat_2[:, ant_ind],
                    )
                    visibility_weights[:, bl_inds, :, use_vis_pol_inds] = 0

    else:  # Flag everything
        flag_antenna_list = []
        for pol_ind in range(N_feed_pols):
            flag_antenna_list.append(antenna_names)
        if visibility_weights is not None:
            visibility_weights[:, :, :, :] = 0

    return flag_antenna_list, visibility_weights


def plot_gains(cal, plot_output_dir, plot_prefix=""):
    """
    Plot gain values. Creates two set of plots for each the gain amplitudes and
    phases. Each figure contains 12 panel, each corresponding to one antenna.
    The feed polarizations are overplotted in each panel.

    Parameters
    ----------
    cal : pyuvdata UVCal object
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    plot_prefix : str
        Optional string to be appended to the start of the file names.
    """

    # Parse strings
    use_plot_prefix = plot_prefix
    if len(plot_prefix) > 0:
        if not use_plot_prefix.endswith("_"):
            use_plot_prefix = f"{use_plot_prefix}_"
    use_plot_output_dir = plot_output_dir
    if plot_output_dir.endswith("/"):
        use_plot_output_dir = use_plot_output_dir[:-1]

    # Plot style parameters
    colors = ["tab:blue", "tab:orange"]
    linewidth = 0.2
    markersize = 0.5
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors[0],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="X",
        ),
        Line2D(
            [0],
            [0],
            color=colors[1],
            marker="o",
            lw=linewidth,
            markersize=markersize,
            label="Y",
        ),
    ]

    ant_names = np.sort(cal.antenna_names)
    freq_axis_mhz = cal.freq_array[0, :] / 1e6

    # Plot amplitudes
    subplot_ind = 0
    plot_ind = 1
    for name in ant_names:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(cal.antenna_names == name)[0][0]
        for pol_ind in range(cal.Njones):
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.abs(cal.gain_array[ant_ind, 0, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                label=(["X", "Y"])[pol_ind],
                color=colors[pol_ind],
            )
        ax.flat[subplot_ind].set_ylim([0, np.nanmax(np.abs(cal.gain_array))])
        ax.flat[subplot_ind].set_xlim([np.min(freq_axis_mhz), np.max(freq_axis_mhz)])
        ax.flat[subplot_ind].set_title(name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or name == ant_names[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Amplitude")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                f"{use_plot_output_dir}/{use_plot_prefix}gain_amp_{plot_ind:02d}.png",
                dpi=600,
            )
            plt.close()
            subplot_ind = 0
            plot_ind += 1

    # Plot phases
    subplot_ind = 0
    plot_ind = 1
    for name in ant_names:
        if subplot_ind == 0:
            fig, ax = plt.subplots(
                nrows=3, ncols=4, figsize=(10, 8), sharex=True, sharey=True
            )
        ant_ind = np.where(cal.antenna_names == name)[0][0]
        for pol_ind in range(cal.Njones):
            ax.flat[subplot_ind].plot(
                freq_axis_mhz,
                np.angle(cal.gain_array[ant_ind, 0, :, 0, pol_ind]),
                "-o",
                linewidth=linewidth,
                markersize=markersize,
                label=(["X", "Y"])[pol_ind],
                color=colors[pol_ind],
            )
        ax.flat[subplot_ind].set_ylim([-np.pi, np.pi])
        ax.flat[subplot_ind].set_xlim([np.min(freq_axis_mhz), np.max(freq_axis_mhz)])
        ax.flat[subplot_ind].set_title(name)
        subplot_ind += 1
        if subplot_ind == len(ax.flat) or name == ant_names[-1]:
            fig.supxlabel("Frequency (MHz)")
            fig.supylabel("Gain Amplitude")
            plt.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.04, 0),
                loc="lower left",
                frameon=False,
            )
            plt.tight_layout()
            plt.savefig(
                f"{use_plot_output_dir}/{use_plot_prefix}gain_phase_{plot_ind:02d}.png",
                dpi=600,
            )
            plt.close()
            subplot_ind = 0
            plot_ind += 1
