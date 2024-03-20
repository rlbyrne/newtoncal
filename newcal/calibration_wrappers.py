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
    antenna_numbers : array of int
        Shape (Nants,). Ordering matches the ordering of the gains attribute.
    antenna_positions : array of float
        Shape (Nants, 3,). Units meters, relative to telescope location.
    channel_width : float
        Width of frequency channels in Hz.
    freq_array : array of float
        Shape (Nfreqs,). Units Hz.
    integration_time : float
        Length of integration in seconds.
    time : float
        Time of observation in Julian Date.
    telescope_name : str
    lst : str
        Local sidereal time (LST), in radians.
    telescope_location : array of float
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
        self.antenna_numbers = None
        self.antenna_positions = None
        self.channel_width = None
        self.freq_array = None
        self.integration_time = None
        self.time = None
        self.telescope_name = None
        self.lst = None
        self.telescope_location = None
        self.lambda_val = None

    def set_gains_from_calfile(self, calfile):
        """
        Use a pyuvdata-formatted calfits file to set gains.

        Parameters
        ----------
        calfile : str
            Path to a pyuvdata-formatted calfits file.
        """

        uvcal = pyuvdata.UVCal()
        uvcal.read_calfits(calfile)

        uvcal.reorder_freqs(channel_order="freq")
        uvcal.reorder_jones()
        use_gains = np.mean(
            uvcal.gain_array[:, 0, :, :, :], axis=2
        )  # Average over times

        cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
        cal_ant_inds = np.array(
            [list(cal_ant_names).index(name) for name in self.antenna_names]
        )

        if self.feed_polarization_array is None:
            self.feed_polarization_array = uvcal.jones_array
        pol_inds = np.array(
            [
                np.where(uvcal.jones_array == feed_pol)[0]
                for feed_pol in feed_polarization_array
            ]
        )

        self.gains = use_gains[cal_ant_inds, :, pol_inds]

    def load_data(
        self,
        data,
        model,
        gain_init_calfile=None,
        gain_init_to_vis_ratio=True,
        gain_init_stddev=0.0,
        N_feed_pols=2,
        feed_polarization_array=None,
        min_cal_baseline_m=None,
        max_cal_baseline_m=None,
        min_cal_baseline_lambda=None,
        max_cal_baseline_lambda=None,
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
        min_cal_baseline_m : float or None
            Minimum baseline length, in meters, to use in calibration. If both
            min_cal_baseline_m and min_cal_baseline_lambda are None, arbitrarily
            short baselines are used. Default None.
        max_cal_baseline_m : float or None
            Maximum baseline length, in meters, to use in calibration. If both
            max_cal_baseline_m and max_cal_baseline_lambda are None, arbitrarily
            long baselines are used. Default None.
        min_cal_baseline_lambda : float or None
            Minimum baseline length, in wavelengths, to use in calibration. If
            both min_cal_baseline_m and min_cal_baseline_lambda are None,
            arbitrarily short baselines are used. Default None.
        max_cal_baseline_lambda : float or None
            Maximum baseline length, in wavelengths, to use in calibration. If
            both max_cal_baseline_m and max_cal_baseline_lambda are None,
            arbitrarily long baselines are used. Default None.
        lambda_val : float
            Weight of the phase regularization term; must be positive. Default
            100.
        """

        # Autocorrelations are not currently supported
        data.select(ant_str="cross")
        model.select(ant_str="cross")

        # Add check to make sure data and model frequencies and times align

        # Downselect baselines
        if (
            (min_cal_baseline_m is not None)
            or (max_cal_baseline_m is not None)
            or (min_cal_baseline_lambda is not None)
            or (max_cal_baseline_lambda is not None)
        ):
            if min_cal_baseline_m is None:
                min_cal_baseline_m = 0.0
            if max_cal_baseline_m is None:
                max_cal_baseline_m = np.inf
            if min_cal_baseline_lambda is None:
                min_cal_baseline_lambda = 0.0
            if max_cal_baseline_lambda is None:
                max_cal_baseline_lambda = np.inf

            max_cal_baseline_m = np.min(
                [
                    max_cal_baseline_lambda * 3e8 / np.min(data.freq_array),
                    max_cal_baseline_m,
                ]
            )
            min_cal_baseline_m = np.max(
                [
                    min_cal_baseline_lambda * 3e8 / np.max(data.freq_array),
                    min_cal_baseline_m,
                ]
            )

            data_baseline_lengths_m = np.sqrt(np.sum(data.uvw_array**2.0, axis=1))
            data_use_baselines = np.where(
                (data_baseline_lengths_m >= min_cal_baseline_m)
                & (data_baseline_lengths_m <= max_cal_baseline_m)
            )
            data.select(blt_inds=data_use_baselines)

            model_baseline_lengths_m = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
            model_use_baselines = np.where(
                (model_baseline_lengths_m >= min_cal_baseline_m)
                & (model_baseline_lengths_m <= max_cal_baseline_m)
            )
            model.select(blt_inds=model_use_baselines)

        self.Nants = data.Nants_data
        self.Nbls = data.Nbls
        self.Ntimes = data.Ntimes
        self.Nfreqs = data.Nfreqs
        self.N_vis_pols = data.Npols

        # Format visibilities
        self.data_visibilities = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=complex,
        )
        self.model_visibilities = np.zeros(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=complex,
        )
        flag_array = np.zeros(
            (self.Ntimes, self.Nbls, self.Nfreqs, self.N_vis_pols), dtype=bool
        )
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
            self.model_visibilities[time_ind, :, :, :] = np.squeeze(
                model_copy.data_array, axis=(1,)
            )
            self.data_visibilities[time_ind, :, :, :] = np.squeeze(
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

        if (min_cal_baseline_lambda is not None) or (
            max_cal_baseline_lambda is not None
        ):
            baseline_lengths_m = np.sqrt(
                np.sum(metadata_reference.uvw_array**2.0, axis=1)
            )
            baseline_lengths_lambda = (
                baseline_lengths_m[:, np.newaxis]
                * metadata_reference.freq_array[0, np.newaxis, :]
                / 3e8
            )
            flag_array[
                :,
                np.where(
                    (baseline_lengths_lambda < min_cal_baseline_lambda)
                    & (baseline_lengths_lambda > max_cal_baseline_lambda)
                ),
                :,
            ] = True

        # Create gains expand matrices
        self.gains_exp_mat_1 = np.zeros((self.Nbls, self.Nants), dtype=int)
        self.gains_exp_mat_2 = np.zeros((self.Nbls, self.Nants), dtype=int)
        self.antenna_numbers = np.unique(
            [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
        )
        for baseline in range(metadata_reference.Nbls):
            self.gains_exp_mat_1[
                baseline,
                np.where(
                    self.antenna_numbers == metadata_reference.ant_1_array[baseline]
                ),
            ] = 1
            self.gains_exp_mat_2[
                baseline,
                np.where(
                    self.antenna_numbers == metadata_reference.ant_2_array[baseline]
                ),
            ] = 1

        # Get ordered list of antenna names
        self.antenna_names = np.array(
            [
                np.array(metadata_reference.antenna_names)[
                    np.where(metadata_reference.antenna_numbers == ant_num)[0][0]
                ]
                for ant_num in self.antenna_numbers
            ]
        )
        self.antenna_positions = np.array(
            [
                np.array(metadata_reference.antenna_positions)[
                    np.where(metadata_reference.antenna_numbers == ant_num)[0][0], :
                ]
                for ant_num in self.antenna_numbers
            ]
        )

        # Get polarization ordering
        self.vis_polarization_array = metadata_reference.polarization_array

        # Initialize gains
        if N_feed_pols is None:
            self.N_feed_pols = 2
        else:
            self.N_feed_pols = N_feed_pols
        if feed_polarization_array is None:
            self.feed_polarization_array = np.array([-5, -6])[: self.N_feed_pols]
        else:
            self.feed_polarization_array = feed_polarization_array
        if gain_init_calfile is None:
            self.gains = np.ones(
                (
                    self.Nants,
                    self.Nfreqs,
                    self.N_feed_pols,
                ),
                dtype=complex,
            )
            if gain_init_to_vis_ratio:  # Use mean ratio of visibility amplitudes
                vis_amp_ratio = np.abs(self.model_visibilities) / np.abs(
                    self.data_visibilities
                )
                vis_amp_ratio[np.where(self.data_visibilities == 0.0)] = np.nan
                self.gains[:, :, :] = np.sqrt(np.nanmedian(vis_amp_ratio))
        else:  # Initialize from file
            self.set_gains_from_calfile(gain_init_calfile)
            # Capture nan-ed gains as flags
            for feed_pol_ind, feed_pol in enumerate(self.feed_polarization_array):
                nan_gains = np.where(~np.isfinite(self.gains[:, :, feed_pol_ind]))
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
                            self.gains_exp_mat_1[:, nan_gains[0][flag_ind]],
                            self.gains_exp_mat_2[:, nan_gains[0][flag_ind]],
                        )
                        flag_freq = nan_gains[1][flag_ind]
                        for flag_pol in flag_pols:
                            flag_array[
                                :,
                                flag_bls,
                                flag_freq,
                                flag_pol,
                            ] = True
                    self.gains[
                        nan_gains, feed_pol_ind
                    ] = 0.0  # Nans in the gains produce matrix multiplication errors, set to zero

        # Grab other metadata from uvfits
        self.channel_width = metadata_reference.channel_width
        self.freq_array = np.reshape(metadata_reference.freq_array, (self.Nfreqs))
        self.integration_time = np.mean(metadata_reference.integration_time)
        self.time = np.mean(metadata_reference.time_array)
        self.telescope_name = metadata_reference.telescope_name
        self.lst = np.mean(metadata_reference.lst_array)
        self.telescope_location = metadata_reference.telescope_location

        # Free memory
        metadata_reference = None

        # Random perturbation of initial gains
        if gain_init_stddev != 0.0:
            self.gains += np.random.normal(
                0.0,
                gain_init_stddev,
                size=(
                    self.Nants,
                    self.Nfreqs,
                    self.N_feed_pols,
                ),
            ) + 1.0j * np.random.normal(
                0.0,
                gain_init_stddev,
                size=(
                    self.Nants,
                    self.Nfreqs,
                    self.N_feed_pols,
                ),
            )

        # Define visibility weights
        self.visibility_weights = np.ones(
            (
                self.Ntimes,
                self.Nbls,
                self.Nfreqs,
                self.N_vis_pols,
            ),
            dtype=float,
        )
        if np.max(flag_array):  # Apply flagging
            self.visibility_weights[np.where(flag_array)] = 0.0

        self.lambda_val = lambda_val

    def convert_to_uvcal(self):
        """
        Generate a pyuvdata UVCal object
        """

        uvcal = pyuvdata.UVCal()
        uvcal.Nants_data = self.Nants
        uvcal.Nants_telescope = self.Nants
        uvcal.Nfreqs = self.Nfreqs
        uvcal.Njones = self.N_feed_pols
        uvcal.Nspws = 1
        uvcal.Ntimes = 1
        uvcal.antenna_names = self.antenna_names
        uvcal.ant_array = self.antenna_numbers
        uvcal.antenna_numbers = self.antenna_numbers
        uvcal.antenna_positions = self.antenna_positions
        uvcal.cal_style = "sky"
        uvcal.cal_type = "gain"
        uvcal.channel_width = self.channel_width
        uvcal.freq_array = self.freq_array[np.newaxis, :]
        uvcal.gain_convention = "multiply"
        uvcal.history = "calibrated with newcal"
        uvcal.integration_time = self.integration_time
        uvcal.jones_array = self.feed_polarization_array
        uvcal.spw_array = np.array([0])
        uvcal.telescope_name = self.telescope_name
        uvcal.lst_array = np.array([self.lst])
        uvcal.telescope_location = self.telescope_location
        uvcal.time_array = np.array([self.time])
        uvcal.time_range = np.array([self.time, self.time])
        uvcal.x_orientation = "east"
        uvcal.gain_array = self.gains[:, np.newaxis, :, np.newaxis, :]
        uvcal.flag_array = (np.isnan(self.gains))[:, np.newaxis, :, np.newaxis, :]
        uvcal.ref_antenna_name = "none"
        uvcal.sky_catalog = ""

        if not uvcal.check():
            print("ERROR: UVCal check failed.")

        return uvcal


def calibration_per_pol(
    caldata_obj,
    xtol=1e-4,
    maxiter=100,
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
    maxiter : int
        Maximum number of iterations for the optimizer. Default 100.
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

    if np.max(caldata_obj.visibility_weights) == 0.0:
        print("ERROR: All data flagged.")
        sys.stdout.flush()
        caldata_obj.gains[:, :, :] = np.nan + 1j * np.nan
    else:

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
            caldata_per_freq.feed_polarization_array = (
                caldata_obj.feed_polarization_array
            )
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
            for freq_ind in range(caldata_obj.Nfreqs):
                args = (
                    caldata_list[freq_ind],
                    xtol,
                    maxiter,
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
                    :, 0, :
                ]
            pool.join()
        else:
            for freq_ind in range(caldata_obj.Nfreqs):
                calibration_optimization.run_calibration_optimization_per_pol_single_freq(
                    caldata_list[freq_ind],
                    xtol,
                    maxiter,
                    verbose,
                )
                caldata_obj.gains[:, freq_ind, :] = caldata_list[freq_ind].gains[
                    :, 0, :
                ]

        if verbose:
            print(
                f"Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - start_time)/60.} minutes"
            )
            sys.stdout.flush()
    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()
