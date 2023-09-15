import numpy as np
import sys
import scipy
import scipy.optimize
import time
import pyuvdata
from newcal import cost_function_calculations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def cost_function_single_pol_wrapper(
    gains_flattened,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val,
):
    """
    Wrapper for function cost_function_single_pol. Reformats the input gains to
    be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. gains_flattened[0:Nants] corresponds to the real
        components of the gains and gains_flattened[Nants:] correponds to the
        imaginary components. Shape (2*Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains = np.reshape(
        gains_flattened,
        (
            2,
            Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    cost = cost_function_calculations.cost_function_single_pol(
        gains,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    return cost


def jacobian_single_pol_wrapper(
    gains_flattened,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val,
):
    """
    Wrapper for function jacobian_single_pol. Reformats the input gains and
    output Jacobian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. gains_flattened[0:Nants] corresponds to the real
        components of the gains and gains_flattened[Nants:] correponds to the
        imaginary components. Shape (2*Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    jac_flattened : array of float
        Jacobian of the cost function, shape (2*Nants,). jac_flattened[0:Nants]
        corresponds to the derivatives with respect to the real part of the
        gains; jac_flattened[Nants:] corresponds to derivatives with respect to
        the imaginary part of the gains.
    """

    gains = np.reshape(
        gains_flattened,
        (
            2,
            Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    jac = cost_function_calculations.jacobian_single_pol(
        gains,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    jac_flattened = np.stack((np.real(jac), np.imag(jac)), axis=0).flatten()
    return jac_flattened


def hessian_single_pol_wrapper(
    gains_flattened,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val,
):
    """
    Wrapper for function hessian_single_pol. Reformats the input gains and
    output Hessian to be compatible with the scipy.optimize.minimize function.

    Parameters
    ----------
    gains_flattened : array of float
        Array of gain values. gains_flattened[0:Nants] corresponds to the real
        components of the gains and gains_flattened[Nants:] correponds to the
        imaginary components. Shape (2*Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities :  array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.

    Returns
    -------
    hess_flattened : array of float
        Hessian of the cost function, shape (2*Nants, 2*Nants,).
    """

    gains = np.reshape(
        gains_flattened,
        (
            2,
            Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    (
        hess_real_real,
        hess_real_imag,
        hess_imag_imag,
    ) = cost_function_calculations.hessian_single_pol(
        gains,
        Nants,
        Nbls,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    hess_flattened = np.full((2 * Nants, 2 * Nants), np.nan, dtype=float)
    hess_flattened[0:Nants, 0:Nants] = hess_real_real
    hess_flattened[Nants:, 0:Nants] = hess_real_imag
    hess_flattened[0:Nants, Nants:] = np.conj(hess_real_imag).T
    hess_flattened[Nants:, Nants:] = hess_imag_imag
    return hess_flattened


def initialize_gains_from_calfile(
    gain_init_calfile,
    Nants,
    Nfreqs,
    N_feed_pols,
    antenna_names,
):
    """
    Extract an array of gains from a pyuvdata-formatted calfits files.

    Parameters
    ----------
    gain_init_calfile : str
        Path to a pyuvdata-formatted calfits file.
    Nants : int
        Number of antennas.
    Nfreqs : int
        Number of frequency channels.
    N_feed_pols : int
        Number of gain polarizations.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains.

    Returns
    -------
    gains_init : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    """

    uvcal = pyuvdata.UVCal()
    uvcal.read_calfits(gain_init_calfile)

    uvcal.reorder_freqs(channel_order="freq")
    uvcal.reorder_jones()
    use_gains = np.mean(uvcal.gain_array[:, 0, :, :, :], axis=3)  # Average over times

    gains_init = np.full(
        (Nants, Nfreqs, N_feed_pols), np.nan + 1j * np.nan, dtype=complex
    )
    cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
    cal_ant_inds = np.array([cal_ant_names.index(name) for name in antenna_names])

    gains_init = use_gains[cal_ant_inds, :, : N_feed_pols + 1]

    return gains_init


def create_uvcal_obj(uvdata, antenna_names, gains=None):
    """
    Generate a pyuvdata UVCal object from gain solutions.

    Parameters
    ----------
    uvdata : pyuvdata UVData object
        Used for metadata reference.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains.
    gains : array of complex or None
        Fit gain values. Shape (Nants, Nfreqs, N_feed_pols,). If None, gains will
        all be set to 1.

    Returns
    -------
    uvcal : pyuvdata UVCal object
    """

    uvcal = pyuvdata.UVCal()
    uvcal.Nants_data = len(antenna_names)
    uvcal.Nants_telescope = uvdata.Nants_telescope
    uvcal.Nfreqs = uvdata.Nfreqs
    if gains is None:
        uvcal.Njones = 2
    else:
        uvcal.Njones = np.shape(gains)[2]
    uvcal.Nspws = 1
    uvcal.Ntimes = 1
    uvcal.ant_array = np.arange(uvcal.Nants_data)
    uvcal.antenna_names = antenna_names
    uvdata_antenna_inds = np.array(
        [(uvdata.antenna_names).index(name) for name in antenna_names]
    )
    uvcal.antenna_numbers = uvdata.antenna_numbers[uvdata_antenna_inds]
    uvcal.antenna_positions = uvdata.antenna_positions[uvdata_antenna_inds, :]
    uvcal.cal_style = "sky"
    uvcal.cal_type = "gain"
    uvcal.channel_width = uvdata.channel_width
    uvcal.freq_array = uvdata.freq_array
    uvcal.gain_convention = "multiply"
    uvcal.history = "calibrated with newcal"
    uvcal.integration_time = np.mean(uvdata.integration_time)
    uvcal.jones_array = np.array([-5, -6, -7, -8])[: uvcal.Njones]
    uvcal.spw_array = uvdata.spw_array
    uvcal.telescope_name = uvdata.telescope_name
    uvcal.time_array = np.array([np.mean(uvdata.time_array)])
    uvcal.time_range = np.array([np.min(uvdata.time_array), np.max(uvdata.time_array)])
    uvcal.x_orientation = "east"
    if gains is None:  # Set all gains to 1
        uvcal.gain_array = np.full(
            (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones),
            1,
            dtype=complex,
        )
    else:
        uvcal.gain_array = gains[:, np.newaxis, :, np.newaxis, :]
    uvcal.flag_array = np.isnan(uvcal.gain_array)
    uvcal.quality_array = np.full(
        (uvcal.Nants_data, uvcal.Nspws, uvcal.Nfreqs, uvcal.Ntimes, uvcal.Njones),
        1.0,
        dtype=float,
    )  # Not supported
    uvcal.ref_antenna_name = "none"
    uvcal.sky_catalog = ""
    uvcal.sky_field = "phase center (RA, Dec): ({}, {})".format(
        np.degrees(np.mean(uvdata.phase_center_app_ra)),
        np.degrees(np.mean(uvdata.phase_center_app_dec)),
    )

    if not uvcal.check():
        print("ERROR: UVCal check failed.")

    return uvcal


def run_calibration_optimization_per_pol_single_freq(
    gains_init,
    Nants,
    Nbls,
    N_feed_pols,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val,
    xtol,
    verbose,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually and the cross-polarization phase is applied from the
    XY and YX visibilities after the fact.

    Parameters
    ----------
    gains_init : array of complex
        Initial guess for the gains. Shape (Nants, N_feed_pols,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    N_feed_pols : int
        Number of feed polarization modes to be fit.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, N_vis_pols,). Polarizations are ordered in the AIPS
        convention: XX, YY, XY, YX.
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, N_vis_pols,). Polarizations are ordered in the AIPS
        convention: XX, YY, XY, YX.
    visibility_weights : array of float
        Shape (Ntimes, Nbls, N_vis_pols,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    lambda_val : float
        Weight of the phase regularization term; must be positive.
    xtol : float
        Accuracy tolerance for optimizer.
    verbose : bool
        Set to True to print optimization outputs.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, N_feed_pols,).
    """

    gains_fit = np.full(
        (
            Nants,
            N_feed_pols,
        ),
        np.nan,
        dtype=complex,
    )
    for pol_ind in range(N_feed_pols):
        if (
            np.max(visibility_weights[:, :, pol_ind]) > 0.0
        ):  # Check if some antennas are fully flagged
            antenna_weights = np.sum(
                np.matmul(
                    gains_exp_mat_1.T,
                    visibility_weights[:, :, pol_ind].T,
                )
                + np.matmul(
                    gains_exp_mat_2.T,
                    visibility_weights[:, :, pol_ind].T,
                ),
                axis=1,
            )
            use_ants = np.where(antenna_weights > 0)[0]
            Nants_use = len(use_ants)
            gains_init_use = gains_init[use_ants, pol_ind]
            gains_init_flattened = np.stack(
                (np.real(gains_init_use), np.imag(gains_init_use)),
                axis=0,
            ).flatten()
            gains_exp_mat_1_use = gains_exp_mat_1[:, use_ants]
            gains_exp_mat_2_use = gains_exp_mat_2[:, use_ants]

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_function_single_pol_wrapper,
                gains_init_flattened,
                args=(
                    Nants_use,
                    Nbls,
                    model_visibilities[:, :, pol_ind],
                    data_visibilities[:, :, pol_ind],
                    visibility_weights[:, :, pol_ind],
                    gains_exp_mat_1_use,
                    gains_exp_mat_2_use,
                    lambda_val,
                ),
                method="Newton-CG",
                jac=jacobian_single_pol_wrapper,
                hess=hessian_single_pol_wrapper,
                options={"disp": verbose, "xtol": xtol, "maxiter": 100},
            )
            end_optimize = time.time()
            if verbose:
                print(result.message)
                print(
                    f"Optimization time: {(end_optimize - start_optimize)/60.} minutes"
                )
            sys.stdout.flush()
            gains_fit_reshaped = np.reshape(result.x, (2, Nants_use))
            gains_fit_single_pol = np.full(Nants, np.nan + 1j * np.nan)
            gains_fit_single_pol[use_ants] = (
                gains_fit_reshaped[0, :] + 1j * gains_fit_reshaped[1, :]
            )

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit_single_pol))),
                np.nanmean(np.cos(np.angle(gains_fit_single_pol))),
            )
            gains_fit_single_pol *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

            gains_fit[:, pol_ind] = gains_fit_single_pol

        else:  # All flagged
            gains_fit[:, pol_ind] = np.full(Nants, np.nan + 1j * np.nan)

    # Constrain crosspol phase
    if N_feed_pols == 2:
        crosspol_phase, gains_fit = cost_function_calculations.set_crosspol_phase(
            gains_fit,
            model_visibilities[:, :, 2:],
            data_visibilities[:, :, 2:],
            visibility_weights[:, :, 2:],
            gains_exp_mat_1,
            gains_exp_mat_2,
            inplace=False,
        )

    return gains_fit


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
):
    """
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

    Returns
    -------
    per_ant_cost_normalized : array of float
        Shape (Nants, N_feed_pols). Encodes the contribution to the cost from
        each antenna and feed, normalized by the number of unflagged baselines.
    """

    per_ant_cost = np.zeros((Nants, N_feed_pols), dtype=float)
    per_ant_baselines = np.zeros((Nants, N_feed_pols), dtype=int)
    for pol_ind in range(N_feed_pols):
        total_visibilities = np.count_nonzero(visibility_weights[:, :, :, pol_ind])
        total_cost = 0.0
        for freq_ind in range(Nfreqs):
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

    per_ant_cost_normalized = per_ant_cost / per_ant_baselines
    return per_ant_cost_normalized


def plot_gains(cal, plot_output_dir, plot_prefix=""):
    """
    Generate a pyuvdata UVCal object from gain solutions.

    Parameters
    ----------
    cal : pyuvdata UVCal object
    plot_output_dir : str
        Path to the directory where the plots will be saved.
    plot_prefix : str
        Optional string to be appended to the start of the file names.
    """

    use_plot_prefix = np.copy(plot_prefix)
    if len(plot_prefix) > 0:
        if not use_plot_prefix.endswith("_"):
            use_plot_prefix = f"{use_plot_prefix}_"

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
                f"{plot_output_dir}/{use_plot_prefix}gain_amp_{plot_ind:02d}.png",
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
                f"{plot_output_dir}/{use_plot_prefix}gain_phase_{plot_ind:02d}.png",
                dpi=600,
            )
            plt.close()
            subplot_ind = 0
            plot_ind += 1
