import numpy as np
import sys
import time
import pyuvdata
import multiprocessing
from newcal import cost_function_calculations
from newcal import calibration_optimization


def uvdata_calibration_setup(
    data,
    model,
    gain_init_calfile=None,
    gain_init_stddev=0.0,
    N_feed_pols=2,
    min_cal_baseline=None,
    max_cal_baseline=None,
):
    """
    Generate the quantities needed for calibration from uvdata objects.

    Parameters
    ----------
    data : pyuvdata UVData object
        Data visibilities to be calibrated.
    model : pyuvdata UVData object
        Model visibilities to be used in calibration. Must have the same
        parameters at data.
    gain_init_calfile : str or None
        Default None. If not None, provides a path to a pyuvdata-formatted
        calfits file containing gains values for calibration initialization. If
        None, all gains are initialized to 1.0.
    gain_init_stddev : float
        Default 0.0. Standard deviation of a complex Gaussian perturbation to
        the initial gains.
    N_feed_pols : int
        Number of gain polarizations. Default 2.
    min_cal_baseline : float or None
        Minimum baseline length, in meters, to use in calibration. If None,
        arbitrarily short baselines are used. Default None.
    max_cal_baseline : float or None
        Maximum baseline length, in meters, to use in calibration. If None,
        arbitrarily long baselines are used. Default None.

    Returns
    -------
    gains_init : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of time intervals.
    Nfreqs : int
        Number of frequency channels.
    N_feed_pols : int
        Number of gain polarizations.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, N_vis_pols,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains.
    """

    # Autocorrelations are not currently supported
    data.select(ant_str="cross")
    model.select(ant_str="cross")

    # Downselect baselines
    if (min_cal_baseline is not None) or (max_cal_baseline is not None):
        if min_cal_baseline is None:
            min_cal_baseline = 0.0
        if max_cal_baseline is None:
            max_cal_baseline = np.inf
        data_baseline_lengths = np.sqrt(np.sum(data.uvw_array**2.0, axis=1))
        data_use_baselines = np.where(
            (data_baseline_lengths >= min_cal_baseline)
            & (data_baseline_lengths <= max_cal_baseline)
        )
        data.select(blt_inds=data_use_baselines)
        model_baseline_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
        model_use_baselines = np.where(
            (model_baseline_lengths >= min_cal_baseline)
            & (model_baseline_lengths <= max_cal_baseline)
        )
        model.select(blt_inds=model_use_baselines)

    Nants = data.Nants_data
    Nbls = data.Nbls
    Ntimes = data.Ntimes
    Nfreqs = data.Nfreqs
    N_vis_pols = data.Npols

    # Format visibilities
    data_visibilities = np.zeros(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            N_vis_pols,
        ),
        dtype=complex,
    )
    model_visibilities = np.zeros(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            N_vis_pols,
        ),
        dtype=complex,
    )
    flag_array = np.zeros((Ntimes, Nbls, Nfreqs, N_vis_pols), dtype=bool)
    for time_ind, time_val in enumerate(np.unique(data.time_array)):
        data_copy = data.copy()
        model_copy = model.copy()
        data_copy.select(times=time_val)
        model_copy.select(times=time_val)
        data_copy.reorder_blts()
        model_copy.reorder_blts()
        data_copy.reorder_pols(order="AIPS")
        model_copy.reorder_pols(order="AIPS")
        data_copy.reorder_freqs(channel_order="freq")
        model_copy.reorder_freqs(channel_order="freq")
        if time_ind == 0:
            metadata_reference = data_copy.copy(metadata_only=True)
        model_visibilities[time_ind, :, :, :] = np.squeeze(
            model_copy.data_array, axis=(1,)
        )
        data_visibilities[time_ind, :, :, :] = np.squeeze(
            data_copy.data_array, axis=(1,)
        )
        flag_array[time_ind, :, :, :] = np.max(
            np.stack(
                [
                    np.squeeze(model_copy.flag_array, axis=(1,)),
                    np.squeeze(data_copy.flag_array, axis=(1,)),
                ]
            ),
            axis=0,
        )

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
    gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
    antenna_list = np.unique(
        [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
    )
    for baseline in range(metadata_reference.Nbls):
        gains_exp_mat_1[
            baseline, np.where(antenna_list == metadata_reference.ant_1_array[baseline])
        ] = 1
        gains_exp_mat_2[
            baseline, np.where(antenna_list == metadata_reference.ant_2_array[baseline])
        ] = 1

    # Get ordered list of antenna names
    antenna_names = np.array(
        [
            np.array(metadata_reference.antenna_names)[
                np.where(metadata_reference.antenna_numbers == ant_num)[0][0]
            ]
            for ant_num in antenna_list
        ]
    )

    # Initialize gains
    if gain_init_calfile is None:  # Use mean ratio of visibility amplitudes
        gains_init = np.ones(
            (
                Nants,
                Nfreqs,
                N_feed_pols,
            ),
            dtype=complex,
        )
        vis_amp_ratio = np.abs(model_visibilities) / np.abs(data_visibilities)
        vis_amp_ratio[np.where(data_visibilities == 0.0)] = np.nan
        gains_init[:, :, :] = np.sqrt(np.nanmean(vis_amp_ratio))
    else:
        gains_init = calibration_optimization.initialize_gains_from_calfile(
            gain_init_calfile,
            Nants,
            Nfreqs,
            N_feed_pols,
            antenna_names,
        )

    if gain_init_stddev != 0.0:
        gains_init += np.random.normal(
            0.0,
            gain_init_stddev,
            size=(
                Nants,
                Nfreqs,
                N_feed_pols,
            ),
        ) + 1.0j * np.random.normal(
            0.0,
            gain_init_stddev,
            size=(
                Nants,
                Nfreqs,
                N_feed_pols,
            ),
        )

    visibility_weights = np.ones(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            N_vis_pols,
        ),
        dtype=float,
    )
    if np.max(flag_array):  # Apply flagging
        visibility_weights[np.where(flag_array)] = 0.0

    return (
        gains_init,
        Nants,
        Nbls,
        Ntimes,
        Nfreqs,
        N_feed_pols,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
        antenna_names,
    )


def calibration_per_pol(
    gains_init,
    Nants,
    Nbls,
    Nfreqs,
    N_feed_pols,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val=100,
    xtol=1e-4,
    parallel=True,
    verbose=False,
    log_file_path=None,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually and the cross-polarization phase is applied from the
    XY and YX visibilities after the fact. Option to parallelize calibration
    across frequency.

    Parameters
    ----------
    gains_init : array of complex
        Initial guess for the gains. Shape (Nants, Nfreqs, N_feed_pols,).
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
    lambda_val : float
        Weight of the phase regularization term; must be positive. Default 100.
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    parallel : bool
        Set to True to parallelize across frequency with multiprocessing.
    verbose : bool
        Set to True to print optimization outputs.
    log_file_path : str or None
        Path to the log file.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, Nfreqs, N_feed_pols,).
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()
    gains_fit = np.full(
        (
            Nants,
            Nfreqs,
            N_feed_pols,
        ),
        np.nan,
        dtype=complex,
    )
    if parallel:
        args_list = []
        for freq_ind in range(Nfreqs):
            args = (
                gains_init[:, freq_ind, :],
                Nants,
                Nbls,
                N_feed_pols,
                model_visibilities[:, :, freq_ind, :],
                data_visibilities[:, :, freq_ind, :],
                visibility_weights[:, :, freq_ind, :],
                gains_exp_mat_1,
                gains_exp_mat_2,
                lambda_val,
                xtol,
                verbose,
            )
            args_list.append(args)
        pool = multiprocessing.Pool()
        result = pool.starmap(
            calibration_optimization.run_calibration_optimization_per_pol_single_freq,
            args_list,
        )
        pool.close()
        for freq_ind in range(Nfreqs):
            gains_fit[:, freq_ind, :] = result[freq_ind]
        pool.join()
    else:
        for freq_ind in range(Nfreqs):
            gains_fit_single_freq = calibration_optimization.run_calibration_optimization_per_pol_single_freq(
                gains_init[:, freq_ind, :],
                Nants,
                Nbls,
                N_feed_pols,
                model_visibilities[:, :, freq_ind, :],
                data_visibilities[:, :, freq_ind, :],
                visibility_weights[:, :, freq_ind, :],
                gains_exp_mat_1,
                gains_exp_mat_2,
                lambda_val,
                xtol,
                verbose,
            )
            gains_fit[:, freq_ind, :] = gains_fit_single_freq

    if verbose:
        print(
            f"Optimization time: {Nfreqs} frequency channels in {(time.time() - start_time)/60.} minutes"
        )
        sys.stdout.flush()
    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return gains_fit
