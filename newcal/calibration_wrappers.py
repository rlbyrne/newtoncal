import numpy as np
import sys
import time
import pyuvdata
import multiprocessing
from newcal import cost_function_calculations
from newcal import calibration_optimization


class CalData:
    """
    Object containing all data and parameters needed for calibration.

    Attributes
    -------
    gains : array of complex
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
    N_vis_pols : int
        Number of visibility polarizations.
    feed_polarization_array : array of int
        Shape (N_feed_pols). Array of polarization integers. Indicates the
        ordering of the polarization axis of the gains. X is -5 and Y is -6.
    vis_polarization_array : array of int
        Shape (N_vis_pols,). Array of polarization integers. Indicates the
        ordering of the polarization axis of the model_visibilities,
        data_visibilities, and visibility_weights. XX is -5, YY is -6, XY is -7,
        and YX is -8.
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
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    lambda_val : float
        Weight of the phase regularization term; must be positive. Default 100.
    """

    def __init__(self):
        self.gains = None
        self.Nants = 0
        self.Nbls = 0
        self.Ntimes = 0
        self.Nfreqs = 0
        self.N_feed_pols = 0
        self.N_vis_pols = 0
        self.feed_polarization_array = None
        self.vis_polarization_array = None
        self.model_visibilities = None
        self.data_visibilities = None
        self.visibility_weights = None
        self.gains_exp_mat_1 = None
        self.gains_exp_mat_2 = None
        self.antenna_names = None
        self.lambda_val = None

    def load_data(
        self,
        data,
        model,
        gain_init_calfile=None,
        gain_init_to_vis_ratio=True,
        gain_init_stddev=0.0,
        N_feed_pols=2,
        feed_polarization_array=None,
        min_cal_baseline=None,
        max_cal_baseline=None,
        lambda_val=100,
    ):
        """
        Format CalData object with parameters from data and model UVData
        objects.

        Parameters
        ----------
        data : pyuvdata UVData object
            Data to be calibrated.
        model : pyuvdata UVData object
            Model visibilities to be used in calibration. Must have the same
            parameters at data.
        gain_init_calfile : str or None
            Default None. If not None, provides a path to a pyuvdata-formatted
            calfits file containing gains values for calibration initialization.
        gain_init_to_vis_ratio : bool
            Used only if gain_init_calfile is None. If True, initializes gains
            to the median ratio between the amplitudes of the model and data
            visibilities. If False, the gains are initialized to 1. Default
            True.
        gain_init_stddev : float
            Default 0.0. Standard deviation of a random complex Gaussian
            perturbation to the initial gains.
        N_feed_pols : int
            Default 2. Number of feed polarizations, equal to the number of gain
            values to be calculated per antenna.
        feed_polarization_array : array of int or None
            Feed polarizations to calibrate. Shape (N_feed_pols,). Options are
            -5 for X or -6 for Y. Default None. If None, feed_polarization_array
            is set to ([-5, -6])[:N_feed_pols].
        min_cal_baseline : float or None
            Minimum baseline length, in meters, to use in calibration. If None,
            arbitrarily short baselines are used. Default None.
        max_cal_baseline : float or None
            Maximum baseline length, in meters, to use in calibration. If None,
            arbitrarily long baselines are used. Default None.
        lambda_val : float
            Weight of the phase regularization term; must be positive. Default
            100.
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

        # Free memory
        data = model = data_copy = model_copy = None

        # Create gains expand matrices
        gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
        gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
        antenna_list = np.unique(
            [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
        )
        for baseline in range(metadata_reference.Nbls):
            gains_exp_mat_1[
                baseline,
                np.where(antenna_list == metadata_reference.ant_1_array[baseline]),
            ] = 1
            gains_exp_mat_2[
                baseline,
                np.where(antenna_list == metadata_reference.ant_2_array[baseline]),
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

        # Get polarization ordering
        vis_polarization_array = metadata_reference.polarization_array

        # Initialize gains
        if gain_init_calfile is None:
            gains_init = np.ones(
                (
                    Nants,
                    Nfreqs,
                    N_feed_pols,
                ),
                dtype=complex,
            )
            if feed_polarization_array is None:
                feed_polarization_array = np.array([-5, -6])[:N_feed_pols]
            if gain_init_to_vis_ratio:  # Use mean ratio of visibility amplitudes
                vis_amp_ratio = np.abs(model_visibilities) / np.abs(data_visibilities)
                vis_amp_ratio[np.where(data_visibilities == 0.0)] = np.nan
                gains_init[:, :, :] = np.sqrt(np.nanmedian(vis_amp_ratio))
        else:  # Initialize from file
            gains_init = initialize_gains_from_calfile(
                gain_init_calfile,
                antenna_names,
                feed_polarization_array,
            )
            # Capture nan-ed gains as flags
            for feed_pol_ind, feed_pol in enumerate(feed_polarization_array):
                nan_gains = np.where(~np.isfinite(gains_init[:, :, feed_pol_ind]))
                if len(nan_gains[0]) > 0:
                    if feed_pol == -5:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -5)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    elif feed_pol == -6:
                        flag_pols = np.where(
                            (metadata_reference.polarization_array == -6)
                            | (metadata_reference.polarization_array == -7)
                            | (metadata_reference.polarization_array == -8)
                        )[0]
                    for flag_ind in range(len(nan_gains[0])):
                        flag_bls = np.logical_or(
                            gains_exp_mat_1[:, nan_gains[0][flag_ind]],
                            gains_exp_mat_2[:, nan_gains[0][flag_ind]],
                        )
                        flag_freq = nan_gains[1][flag_ind]
                        for flag_pol in flag_pols:
                            flag_array[
                                :,
                                flag_bls,
                                flag_freq,
                                flag_pol,
                            ] = True
                    gains_init[
                        nan_gains, feed_pol_ind
                    ] = 0.0  # Nans in the gains produce matrix multiplication errors, set to zero

        # Free memory
        metadata_reference = None

        # Random perturbation of initial gains
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

        # Define visibility weights
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

        self.gains = gains_init
        self.Nants = Nants
        self.Nbls = Nbls
        self.Ntimes = Ntimes
        self.Nfreqs = Nfreqs
        self.N_feed_pols = N_feed_pols
        self.N_vis_pols = N_vis_pols
        self.feed_polarization_array = feed_polarization_array
        self.vis_polarization_array = vis_polarization_array
        self.model_visibilities = model_visibilities
        self.data_visibilities = data_visibilities
        self.visibility_weights = visibility_weights
        self.gains_exp_mat_1 = gains_exp_mat_1
        self.gains_exp_mat_2 = gains_exp_mat_2
        self.antenna_names = antenna_names
        self.lambda_val = lambda_val


def initialize_gains_from_calfile(
    gain_init_calfile,
    antenna_names,
    feed_polarization_array,
):
    """
    Extract an array of gains from a pyuvdata-formatted calfits files.

    Parameters
    ----------
    gain_init_calfile : str
        Path to a pyuvdata-formatted calfits file.
    antenna_names : array of str
        Shape (Nants,). Ordering matches the ordering of the gains.
    feed_polarization_array : array of int or None
        Shape (N_feed_pols,). -5 is X and -6 is Y. If None, uses the
        polarizations from gain_init_calfile.

    Returns
    -------
    gains_init : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    """

    uvcal = pyuvdata.UVCal()
    uvcal.read_calfits(gain_init_calfile)

    uvcal.reorder_freqs(channel_order="freq")
    uvcal.reorder_jones()
    use_gains = np.mean(uvcal.gain_array[:, 0, :, :, :], axis=2)  # Average over times

    cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
    cal_ant_inds = np.array([list(cal_ant_names).index(name) for name in antenna_names])

    if feed_polarization_array is None:
        feed_polarization_array = uvcal.jones_array
    pol_inds = np.array(
        [
            np.where(uvcal.jones_array == feed_pol)[0]
            for feed_pol in feed_polarization_array
        ]
    )

    gains_init = use_gains[cal_ant_inds, :, pol_inds]

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
        Fit gain values. Shape (Nants, Nfreqs, N_feed_pols,). If None, gains
        will all be set to 1.

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


def calibration_per_pol(
    caldata_obj,
    xtol=1e-4,
    parallel=True,
    verbose=False,
    log_file_path=None,
):
    """
    Run calibration per polarization. Function updates the gains attribute of
    the CalData object. Here the XX and YY visibilities are calibrated
    individually and the cross-polarization phase is applied from the XY and YX
    visibilities after the fact. Option to parallelize calibration across
    frequency.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    parallel : bool
        Set to True to parallelize across frequency with multiprocessing.
        Default True.
    verbose : bool
        Set to True to print optimization outputs. Default False.
    log_file_path : str or None
        Path to the log file. Default None.
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    # Expand CalData object into per-frequency objects
    caldata_list = []
    for freq_ind in range(caldata_obj.Nfreqs):
        caldata_per_freq = CalData()
        caldata_per_freq.gains = caldata_obj.gains[:, [freq_ind], :]
        caldata_per_freq.Nants = caldata_obj.Nants
        caldata_per_freq.Nbls = caldata_obj.Nbls
        caldata_per_freq.Ntimes = caldata_obj.Ntimes
        caldata_per_freq.Nfreqs = 1
        caldata_per_freq.N_feed_pols = caldata_obj.N_feed_pols
        caldata_per_freq.feed_polarization_array = caldata_obj.feed_polarization_array
        caldata_per_freq.vis_polarization_array = caldata_obj.vis_polarization_array
        caldata_per_freq.model_visibilities = caldata_obj.model_visibilities[
            :, :, [freq_ind], :
        ]
        caldata_per_freq.data_visibilities = caldata_obj.data_visibilities[
            :, :, [freq_ind], :
        ]
        caldata_per_freq.visibility_weights = caldata_obj.visibility_weights[
            :, :, [freq_ind], :
        ]
        caldata_per_freq.gains_exp_mat_1 = caldata_obj.gains_exp_mat_1
        caldata_per_freq.gains_exp_mat_2 = caldata_obj.gains_exp_mat_2
        caldata_per_freq.antenna_names = caldata_obj.antenna_names
        caldata_per_freq.lambda_val = caldata_obj.lambda_val
        caldata_list.append(caldata_per_freq)

    gains_fit = np.full(
        (
            caldata_obj.Nants,
            caldata_obj.Nfreqs,
            caldata_obj.N_feed_pols,
        ),
        np.nan,
        dtype=complex,
    )
    if parallel:
        args_list = []
        for freq_ind in range(Nfreqs):
            args = (
                caldata_list[freq_ind],
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
        for freq_ind in range(caldata_obj.Nfreqs):
            caldata_obj.gains[:, freq_ind, :] = args_list[freq_ind][0].gains[
                :, freq_ind, :
            ]
        pool.join()
    else:
        for freq_ind in range(caldata_obj.Nfreqs):
            calibration_optimization.run_calibration_optimization_per_pol_single_freq(
                caldata_list[freq_ind],
                xtol,
                verbose,
            )
            caldata_obj.gains[:, freq_ind, :] = caldata_list[freq_ind].gains[
                :, freq_ind, :
            ]

    if verbose:
        print(
            f"Optimization time: {Nfreqs} frequency channels in {(time.time() - start_time)/60.} minutes"
        )
        sys.stdout.flush()
    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()
