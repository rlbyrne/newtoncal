import numpy as np
import sys
import time
import pyuvdata
import multiprocessing
from newcal import calibration_optimization


class CalData:
    """
    Object containing all data and parameters needed for calibration.

    Attributes
    -------
    gains : array of complex
        Shape (Nants, Nfreqs, N_feed_pols,).
    abscal_params : array of float
        Shape (3, Nfreqs, N_feed_pols). abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
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
    uv_array : array of float
        Shape (Nbls, 2,). Baseline positions in the UV plane, units meters.
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
        self.abscal_params = None
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
        self.uv_array = None
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
        uvcal.select(frequencies=self.freq_array, antenna_names=self.antenna_names)
        if self.feed_polarization_array is None:
            self.feed_polarization_array = uvcal.jones_array
        else:
            uvcal.select(jones=self.feed_polarization_array)
        uvcal.reorder_freqs(channel_order="freq")
        uvcal.reorder_jones()
        use_gains = np.mean(
            uvcal.gain_array[:, 0, :, :, :], axis=2
        )  # Average over times

        # Make antenna ordering match
        cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
        cal_ant_inds = np.array(
            [list(cal_ant_names).index(name) for name in self.antenna_names]
        )

        self.gains = use_gains[cal_ant_inds, :, :]

    def load_data(
        self,
        data,
        model,
        gain_init_calfile=None,
        gain_init_to_vis_ratio=True,
        gain_init_stddev=0.0,
        N_feed_pols=None,
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
            Default min(2, N_vis_pols). Number of feed polarizations, equal to
            the number of gain values to be calculated per antenna.
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

        # Grab other metadata from uvfits
        self.channel_width = metadata_reference.channel_width
        self.freq_array = np.reshape(metadata_reference.freq_array, (self.Nfreqs))
        self.integration_time = np.mean(metadata_reference.integration_time)
        self.time = np.mean(metadata_reference.time_array)
        self.telescope_name = metadata_reference.telescope_name
        self.lst = np.mean(metadata_reference.lst_array)
        self.telescope_location = metadata_reference.telescope_location

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

        # Get UV locations
        antpos_ecef = (
            self.antenna_positions + metadata_reference.telescope_location
        )  # Get antennas positions in ECEF
        antpos_enu = pyuvdata.utils.ENU_from_ECEF(
            antpos_ecef, *metadata_reference.telescope_location_lat_lon_alt
        )  # Convert to topocentric (East, North, Up or ENU) coords.
        uvw_array = np.matmul(self.gains_exp_mat_1, antpos_enu) - np.matmul(
            self.gains_exp_mat_2, antpos_enu
        )
        self.uv_array = uvw_array[:, :2]

        # Get polarization ordering
        self.vis_polarization_array = np.array(metadata_reference.polarization_array)

        if N_feed_pols is None:
            self.N_feed_pols = np.min([2, self.N_vis_pols])
        else:
            self.N_feed_pols = N_feed_pols

        if feed_polarization_array is None:
            self.feed_polarization_array = np.array([], dtype=int)
            if (
                (-5 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -5
                )
            if (
                (-6 in self.vis_polarization_array)
                or (-7 in self.vis_polarization_array)
                or (-8 in self.vis_polarization_array)
            ):
                self.feed_polarization_array = np.append(
                    self.feed_polarization_array, -6
                )
            self.feed_polarization_array = self.feed_polarization_array[
                : self.N_feed_pols
            ]
        else:
            self.feed_polarization_array = feed_polarization_array

        # Initialize gains
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
                    self.gains[nan_gains, feed_pol_ind] = (
                        0.0  # Nans in the gains produce matrix multiplication errors, set to zero
                    )

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

        # Initialize abscal parameters
        self.abscal_params = np.zeros((3, self.Nfreqs, self.N_feed_pols), dtype=float)
        self.abscal_params[0, :, :] = 1.0

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

    def expand_in_frequency(self):
        """
        Converts a caldata object into a list of caldata objects each
        corresponding to one frequency.

        Returns
        -------
        caldata_list : list of caldata objects
        """

        caldata_list = []
        for freq_ind in range(self.Nfreqs):
            caldata_per_freq = CalData()
            caldata_per_freq.gains = self.gains[:, [freq_ind], :]
            caldata_per_freq.abscal_params = self.abscal_params[:, [freq_ind], :]
            caldata_per_freq.Nants = self.Nants
            caldata_per_freq.Nbls = self.Nbls
            caldata_per_freq.Ntimes = self.Ntimes
            caldata_per_freq.Nfreqs = 1
            caldata_per_freq.N_feed_pols = self.N_feed_pols
            caldata_per_freq.N_vis_pols = self.N_vis_pols
            caldata_per_freq.feed_polarization_array = self.feed_polarization_array
            caldata_per_freq.vis_polarization_array = self.vis_polarization_array
            caldata_per_freq.model_visibilities = self.model_visibilities[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.data_visibilities = self.data_visibilities[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.visibility_weights = self.visibility_weights[
                :, :, [freq_ind], :
            ]
            caldata_per_freq.gains_exp_mat_1 = self.gains_exp_mat_1
            caldata_per_freq.gains_exp_mat_2 = self.gains_exp_mat_2
            caldata_per_freq.antenna_names = self.antenna_names
            caldata_per_freq.antenna_numbers = self.antenna_numbers
            caldata_per_freq.antenna_positions = self.antenna_positions
            caldata_per_freq.uv_array = self.uv_array
            caldata_per_freq.channel_width = self.channel_width
            caldata_per_freq.freq_array = self.freq_array[[freq_ind]]
            caldata_per_freq.integration_time = self.integration_time
            caldata_per_freq.time = self.time
            caldata_per_freq.telescope_name = self.telescope_name
            caldata_per_freq.lst = self.lst
            caldata_per_freq.telescope_location = self.telescope_location
            caldata_per_freq.lambda_val = self.lambda_val
            caldata_list.append(caldata_per_freq)

        return caldata_list

    def expand_in_polarization(self):
        """
        Converts a caldata object into a list of caldata objects each
        corresponding to one feed polarization. List does not include
        cross-polarization visibilities.

        Returns
        -------
        caldata_list : list of caldata objects
        """

        caldata_list = []
        for feed_pol_ind, pol in enumerate(self.feed_polarization_array):
            caldata_per_pol = CalData()
            sky_pol_ind = np.where(self.vis_polarization_array == pol)[0][0]
            caldata_per_pol.gains = self.gains[:, :, [feed_pol_ind]]
            caldata_per_pol.abscal_params = self.abscal_params[:, :, [feed_pol_ind]]
            caldata_per_pol.Nants = self.Nants
            caldata_per_pol.Nbls = self.Nbls
            caldata_per_pol.Ntimes = self.Ntimes
            caldata_per_pol.Nfreqs = self.Nfreqs
            caldata_per_pol.N_feed_pols = 1
            caldata_per_pol.N_vis_pols = 1
            caldata_per_pol.feed_polarization_array = self.feed_polarization_array[
                [feed_pol_ind]
            ]
            caldata_per_pol.vis_polarization_array = self.vis_polarization_array[
                [sky_pol_ind]
            ]
            caldata_per_pol.model_visibilities = self.model_visibilities[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.data_visibilities = self.data_visibilities[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.visibility_weights = self.visibility_weights[
                :, :, :, [sky_pol_ind]
            ]
            caldata_per_pol.gains_exp_mat_1 = self.gains_exp_mat_1
            caldata_per_pol.gains_exp_mat_2 = self.gains_exp_mat_2
            caldata_per_pol.antenna_names = self.antenna_names
            caldata_per_pol.antenna_numbers = self.antenna_numbers
            caldata_per_pol.antenna_positions = self.antenna_positions
            caldata_per_pol.uv_array = self.uv_array
            caldata_per_pol.channel_width = self.channel_width
            caldata_per_pol.freq_array = self.freq_array
            caldata_per_pol.integration_time = self.integration_time
            caldata_per_pol.time = self.time
            caldata_per_pol.telescope_name = self.telescope_name
            caldata_per_pol.lst = self.lst
            caldata_per_pol.telescope_location = self.telescope_location
            caldata_per_pol.lambda_val = self.lambda_val

            """
            if np.max(caldata_per_pol.visibility_weights) > 0.0:
                # Check if some antennas are fully flagged
                antenna_weights = np.sum(
                    np.matmul(
                        caldata_per_pol.gains_exp_mat_1.T,
                        caldata_per_pol.visibility_weights[:, :, 0, 0].T,
                    )
                    + np.matmul(
                        caldata_per_pol.gains_exp_mat_2.T,
                        caldata_per_pol.visibility_weights[:, :, 0, 0].T,
                    ),
                    axis=1,
                )
                use_ants = np.where(antenna_weights > 0)[0]
                if len(use_ants) != caldata_per_pol.Nants:
                    caldata_per_pol.gains = caldata_per_pol.gains[use_ants, :, :]
                    caldata_per_pol.Nants = len(use_ants)
                    caldata_per_pol.gains_exp_mat_1 = caldata_per_pol.gains_exp_mat_1[
                        :, use_ants
                    ]
                    caldata_per_pol.gains_exp_mat_2 = caldata_per_pol.gains_exp_mat_2[
                        :, use_ants
                    ]
                    caldata_per_pol.antenna_names = caldata_per_pol.antenna_names[
                        use_ants
                    ]
            """

            caldata_list.append(caldata_per_pol)

        return caldata_list

    def convert_to_uvcal(self):
        """
        Generate a pyuvdata UVCal object.

        Returns
        -------
        uvcal : pyuvdata UVCal object
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
        uvcal.time_range = np.array([self.time, self.time])[np.newaxis, :]
        uvcal.lst_range = np.array([self.lst, self.lst])[np.newaxis, :]
        uvcal.x_orientation = "east"
        uvcal.gain_array = self.gains[:, np.newaxis, :, np.newaxis, :]
        uvcal.ref_antenna_name = "none"
        uvcal.sky_catalog = ""

        # Get flags from nan-ed gains and zeroed weights
        uvcal.flag_array = (np.isnan(self.gains))[:, np.newaxis, :, np.newaxis, :]
        # Flag antennas
        antenna_weights = np.sum(
            np.matmul(
                self.gains_exp_mat_1.T,
                self.visibility_weights[:, :, 0, 0].T,
            )
            + np.matmul(
                self.gains_exp_mat_2.T,
                self.visibility_weights[:, :, 0, 0].T,
            ),
            axis=1,
        )
        uvcal.flag_array[np.where(antenna_weights == 0)[0], :, :, :, :] = True
        # Flag frequencies
        freq_weights = np.sum(self.visibility_weights, axis=(0, 1, 3))
        uvcal.flag_array[:, :, np.where(freq_weights == 0)[0], :, :] = True

        uvcal.use_future_array_shapes()

        if not uvcal.check():
            print("ERROR: UVCal check failed.")

        return uvcal


def apply_abscal(uvdata, abscal_params, feed_polarization_array, inplace=False):
    """
    Apply absolute calibration solutions to data.

    Parameters
    ----------
    uvdata : pyuvdata UVData object
        pyuvdata UVData object containing the data.
    abscal_params : array of float
        Shape (3, Nfreqs, N_feed_pols). abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
    feed_polarization_array : array of int
        Shape (N_feed_pols). Array of polarization integers. Indicates the
        ordering of the polarization axis of the gains. X is -5 and Y is -6.
    inplace : bool
        If True, updates uvdata. If False, returns a new UVData object.

    Returns
    -------
    uvdata_new : pyuvdata UVData object
        Returned only if inplace is False.
    """

    if not inplace:
        uvdata_new = uvdata.copy()

    # Get antenna locations
    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((uvdata.Nblts, len(uvdata.antenna_numbers)), dtype=int)
    gains_exp_mat_2 = np.zeros((uvdata.Nblts, len(uvdata.antenna_numbers)), dtype=int)
    for baseline in range(uvdata.Nblts):
        gains_exp_mat_1[
            baseline,
            np.where(uvdata.antenna_numbers == uvdata.ant_1_array[baseline]),
        ] = 1
        gains_exp_mat_2[
            baseline,
            np.where(uvdata.antenna_numbers == uvdata.ant_2_array[baseline]),
        ] = 1
    antpos_ecef = (
        uvdata.antenna_positions + uvdata.telescope_location
    )  # Get antennas positions in ECEF
    antpos_enu = pyuvdata.utils.ENU_from_ECEF(
        antpos_ecef, *uvdata.telescope_location_lat_lon_alt
    )  # Convert to topocentric (East, North, Up or ENU) coords.
    antpos_en = antpos_enu[:, :2]
    ant1_positions = np.matmul(gains_exp_mat_1, antpos_en)
    ant2_positions = np.matmul(gains_exp_mat_2, antpos_en)

    for vis_pol_ind, vis_pol in enumerate(uvdata.polarization_array):
        if vis_pol == -5:
            pol1 = pol2 = np.where(feed_polarization_array == -5)[0][0]
        elif vis_pol == -6:
            pol1 = pol2 = np.where(feed_polarization_array == -6)[0][0]
        elif vis_pol == -7:
            pol1 = np.where(feed_polarization_array == -5)[0][0]
            pol2 = np.where(feed_polarization_array == -6)[0][0]
        elif vis_pol == -8:
            pol1 = np.where(feed_polarization_array == -6)[0][0]
            pol2 = np.where(feed_polarization_array == -5)[0][0]
        else:
            print(f"ERROR: Polarization {vis_pol} not recognized.")
            sys.exit(1)

        amp_term = (
            abscal_params[0, :, pol1] * abscal_params[0, :, pol2]
        )  # Shape (Nfreqs,)
        phase_correction = np.exp(
            1j
            * (
                abscal_params[1, np.newaxis, :, pol1] * ant1_positions[:, np.newaxis, 0]
                - abscal_params[1, np.newaxis, :, pol2]
                * ant2_positions[:, np.newaxis, 0]
                + abscal_params[2, np.newaxis, :, pol1]
                * ant1_positions[:, np.newaxis, 1]
                - abscal_params[2, np.newaxis, :, pol2]
                * ant2_positions[:, np.newaxis, 1]
            )
        )  # Shape (Nbls, Nfreqs,)

        if inplace:
            uvdata.data_array[:, :, :, vis_pol_ind] *= (
                amp_term[np.newaxis, np.newaxis, :] * phase_correction[:, np.newaxis, :]
            )
        else:
            uvdata_new.data_array[:, :, :, vis_pol_ind] *= (
                amp_term[np.newaxis, np.newaxis, :] * phase_correction[:, np.newaxis, :]
            )

    if not inplace:
        return uvdata_new


def absolute_calibration(
    data_file_path,
    model_file_path,
    data_use_column="DATA",
    model_use_column="MODEL_DATA",
    N_feed_pols=None,
    feed_polarization_array=None,
    min_cal_baseline_m=None,
    max_cal_baseline_m=None,
    min_cal_baseline_lambda=None,
    max_cal_baseline_lambda=None,
    xtol=1e-4,
    maxiter=100,
    verbose=False,
    log_file_path=None,
):
    """
    Top-level wrapper for running absolute calibration ("abscal").

    Parameters
    ----------
    data_file_path : str
        Path to the ms or uvfits file containing the relatively calibrated
        data visibilities.
    model_file_path : str
        Path to the ms or uvfits file containing the model visibilities.
    data_use_column : str
        Column in an ms file to use for the data visibilities. Used only if
        data_file_path points to an ms file. Default "DATA".
    model_use_column : str
        Column in an ms file to use for the model visibilities. Used only if
        data_file_path points to an ms file. Default "MODEL_DATA".
    N_feed_pols : int
        Default min(2, N_vis_pols). Number of feed polarizations, equal to
        the number of gain values to be calculated per antenna.
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
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    maxiter : int
        Maximum number of iterations for the optimizer. Default 100.
    verbose : bool
        Set to True to print optimization outputs. Default False.
    log_file_path : str or None
        Path to the log file. Default None.
    Returns
    -------
    abscal_params : array of float
        Shape (3, Nfreqs, N_feed_pols). abscal_params[0, :, :] are the overall amplitudes,
        abscal_params[1, :, :] are the x-phase gradients in units 1/m, and abscal_params[2, :, :]
        are the y-phase gradients in units 1/m.
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    if verbose:
        print("Reading data...")
        sys.stdout.flush()
        data_read_start_time = time.time()

    # Read data
    data = pyuvdata.UVData()
    if data_file_path.endswith(".ms"):
        data.read_ms(data_file_path, data_column=data_use_column)
    elif data_file_path.endswith(".uvfits"):
        data.read_uvfits(data_file_path)
    else:
        print(f"ERROR: Unsupported file type for data file {data_file_path}. Exiting.")
        sys.exit(1)
    model = pyuvdata.UVData()
    if model_file_path.endswith(".ms"):
        model.read_ms(model_file_path, data_column=model_use_column)
    elif model_file_path.endswith(".uvfits"):
        model.read_uvfits(model_file_path)
    else:
        print(
            f"ERROR: Unsupported file type for model file {model_file_path}. Exiting."
        )

    if verbose:
        print(
            f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
        )
        print("Formatting data...")
        sys.stdout.flush()
        data_format_start_time = time.time()

    caldata_obj = CalData()
    caldata_obj.load_data(
        data,
        model,
        N_feed_pols=N_feed_pols,
        feed_polarization_array=feed_polarization_array,
        min_cal_baseline_m=min_cal_baseline_m,
        max_cal_baseline_m=max_cal_baseline_m,
        min_cal_baseline_lambda=min_cal_baseline_lambda,
        max_cal_baseline_lambda=max_cal_baseline_lambda,
    )

    if verbose:
        print(
            f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Running calibration optimization...")
        sys.stdout.flush()

    # Expand CalData object into per-frequency objects
    caldata_list = caldata_obj.expand_in_frequency()

    optimization_start_time = time.time()

    for freq_ind in range(caldata_obj.Nfreqs):
        calibration_optimization.run_abscal_optimization_single_freq(
            caldata_list[freq_ind],
            xtol,
            maxiter,
            verbose=verbose,
        )
        caldata_obj.abscal_params[:, [freq_ind], :] = caldata_list[
            freq_ind
        ].abscal_params[:, [0], :]

    if verbose:
        print(
            f"Done. Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes"
        )
        print(f"Total processing time {(time.time() - start_time)/60.} minutes.")
        sys.stdout.flush()

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return caldata_obj.abscal_params


def calibrate_caldata_per_pol(
    caldata_obj,
    xtol=1e-4,
    maxiter=100,
    get_crosspol_phase=True,
    parallel=True,
    verbose=False,
    pool=None,
):
    """
    Run calibration per polarization. Updates the gains attribute of caldata_obj
    with calibrated values. Here the XX and YY visibilities are calibrated
    individually and the cross-polarization phase is applied fromthe XY and YX
    visibilities after the fact. Option to parallelize calibration across frequency.

    Parameters
    ----------
    caldata_obj : CalData
        CalData object containing the data and model visibilities for calibration.
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    maxiter : int
        Maximum number of iterations for the optimizer. Default 100.
    get_crosspol_phase : bool
        If True, crosspol phase is calculated. Default True.
    parallel : bool
        Set to True to parallelize across frequency with multiprocessing.
        Default True if Nfreqs > 1.
    verbose : bool
        Set to True to print optimization outputs. Default False.
    pool : multiprocessing.pool.Pool or None
        Pool for multiprocessing. Must not be None if parallel=True.
    """

    if np.max(caldata_obj.visibility_weights) == 0.0:
        print("ERROR: All data flagged.")
        sys.stdout.flush()
        caldata_obj.gains[:, :, :] = np.nan + 1j * np.nan
    else:

        # Expand CalData object into per-frequency objects
        caldata_list = caldata_obj.expand_in_frequency()

        optimization_start_time = time.time()

        if parallel:
            args_list = []
            for freq_ind in range(caldata_obj.Nfreqs):
                args = (
                    caldata_list[freq_ind],
                    xtol,
                    maxiter,
                    verbose,
                    get_crosspol_phase,
                    True,
                )
                args_list.append(args)
            result = pool.starmap(
                calibration_optimization.run_calibration_optimization_per_pol_single_freq,
                args_list,
            )
            pool.close()
            for freq_ind in range(caldata_obj.Nfreqs):
                caldata_obj.gains[:, [freq_ind], :] = result[freq_ind]
            pool.join()
        else:
            for freq_ind in range(caldata_obj.Nfreqs):
                calibration_optimization.run_calibration_optimization_per_pol_single_freq(
                    caldata_list[freq_ind],
                    xtol,
                    maxiter,
                    verbose=verbose,
                    get_crosspol_phase=get_crosspol_phase,
                )
                caldata_obj.gains[:, [freq_ind], :] = caldata_list[freq_ind].gains[
                    :, [0], :
                ]

        if verbose:
            print(
                f"Done. Optimization time: {caldata_obj.Nfreqs} frequency channels in {(time.time() - optimization_start_time)/60.} minutes"
            )
            sys.stdout.flush()


def calibration_per_pol(
    data_file_path,
    model_file_path,
    data_use_column="DATA",
    model_use_column="MODEL_DATA",
    gain_init_calfile=None,
    gain_init_to_vis_ratio=True,
    gain_init_stddev=0.0,
    N_feed_pols=None,
    feed_polarization_array=None,
    min_cal_baseline_m=None,
    max_cal_baseline_m=None,
    min_cal_baseline_lambda=None,
    max_cal_baseline_lambda=None,
    lambda_val=100,
    xtol=1e-4,
    maxiter=100,
    get_crosspol_phase=True,
    parallel=True,
    max_processes=40,
    verbose=False,
    log_file_path=None,
):
    """
    Top-level wrapper for running calibration per polarization. Function creates
    a CalData object, updates the gains attribute, and returns a pyuvdata UVCal
    object containing the calibration solutions. Here the XX and YY visibilities
    are calibrated individually and the cross-polarization phase is applied from
    the XY and YX visibilities after the fact. Option to parallelize calibration
    across frequency.

    Parameters
    ----------
    data_file_path : str
        Path to the ms or uvfits file containing the data visibilities.
    model_file_path : str
        Path to the ms or uvfits file containing the model visibilities.
    data_use_column : str
        Column in an ms file to use for the data visibilities. Used only if
        data_file_path points to an ms file. Default "DATA".
    model_use_column : str
        Column in an ms file to use for the model visibilities. Used only if
        data_file_path points to an ms file. Default "MODEL_DATA".
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
        Default min(2, N_vis_pols). Number of feed polarizations, equal to
        the number of gain values to be calculated per antenna.
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
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    maxiter : int
        Maximum number of iterations for the optimizer. Default 100.
    get_crosspol_phase : bool
        If True, crosspol phase is calculated. Default True.
    parallel : bool
        Set to True to parallelize across frequency with multiprocessing.
        Default True if Nfreqs > 1.
    max_processes : int or None
        Maximum number of multithreaded processes to use. Applicable only if
        parallel is True. If None, uses the multiprocessing default. Default 40.
    verbose : bool
        Set to True to print optimization outputs. Default False.
    log_file_path : str or None
        Path to the log file. Default None.
    Returns
    -------
    uvcal : pyuvdata UVCal object
    """

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start_time = time.time()

    if parallel:  # Start multiprocessing pool
        if max_processes is None:
            pool = multiprocessing.Pool()
        else:
            pool = multiprocessing.Pool(processes=max_processes)

    if verbose:
        print("Reading data...")
        sys.stdout.flush()
        data_read_start_time = time.time()

    # Read data
    data = pyuvdata.UVData()
    if data_file_path.endswith(".ms"):
        data.read_ms(data_file_path, data_column=data_use_column)
    elif data_file_path.endswith(".uvfits"):
        data.read_uvfits(data_file_path)
    else:
        print(f"ERROR: Unsupported file type for data file {data_file_path}. Exiting.")
        sys.exit(1)
    model = pyuvdata.UVData()
    if model_file_path.endswith(".ms"):
        model.read_ms(model_file_path, data_column=model_use_column)
    elif model_file_path.endswith(".uvfits"):
        model.read_uvfits(model_file_path)
    else:
        print(
            f"ERROR: Unsupported file type for model file {model_file_path}. Exiting."
        )

    if verbose:
        print(
            f"Done. Data read time {(time.time() - data_read_start_time)/60.} minutes."
        )
        print("Formatting data...")
        sys.stdout.flush()
        data_format_start_time = time.time()

    caldata_obj = CalData()
    caldata_obj.load_data(
        data,
        model,
        gain_init_calfile=gain_init_calfile,
        gain_init_to_vis_ratio=gain_init_to_vis_ratio,
        gain_init_stddev=gain_init_stddev,
        N_feed_pols=N_feed_pols,
        feed_polarization_array=feed_polarization_array,
        min_cal_baseline_m=min_cal_baseline_m,
        max_cal_baseline_m=max_cal_baseline_m,
        min_cal_baseline_lambda=min_cal_baseline_lambda,
        max_cal_baseline_lambda=max_cal_baseline_lambda,
        lambda_val=lambda_val,
    )

    if caldata_obj.Nfreqs < 2:
        parallel = False
        pool.close()
        pool.join()

    if verbose:
        print(
            f"Done. Data formatting time {(time.time() - data_format_start_time)/60.} minutes."
        )
        print("Running calibration optimization...")
        sys.stdout.flush()

    calibrate_caldata_per_pol(
        caldata_obj,
        xtol=xtol,
        maxiter=maxiter,
        get_crosspol_phase=get_crosspol_phase,
        parallel=parallel,
        verbose=verbose,
        pool=pool,
    )

    # Convert to UVCal object
    uvcal = caldata_obj.convert_to_uvcal()

    if verbose:
        print(f"Total processing time {(time.time() - start_time)/60.} minutes.")
        sys.stdout.flush()

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()

    return uvcal
