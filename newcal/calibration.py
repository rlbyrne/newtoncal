import numpy as np
import sys
import scipy
import scipy.optimize
import time
import pyuvdata
from newcal import cost_function_calculations


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
        Jacobian of the cost function, shape (2*Nants, 2*Nants,). jac_flattened[0:Nants]
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
    hess_flattened[0:Nants, Nants:] = hess_real_imag
    hess_flattened[Nants:, Nants:] = hess_imag_imag
    return hess_flattened


def initialize_gains_from_calfile(
    gain_init_calfile,
    Nants,
    Nfreqs,
    antenna_list,
    antenna_names,
    time_ind=0,
    pol_ind=0,
):
    """
    Need to edit this file to include all polarizations
    """

    uvcal = pyuvdata.UVCal()
    uvcal.read_calfits(gain_init_calfile)
    gains_init = np.ones((Nants, Nfreqs), dtype=complex)
    cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
    for ind, ant in enumerate(antenna_list):
        ant_name = antenna_names[ant]
        cal_ant_ind = np.where(cal_ant_names == ant_name)[0][0]
        gains_init[ind, :] = uvcal.gain_array[cal_ant_ind, 0, :, time_ind, pol_ind]

    return gains_init


def uvdata_calibration_setup(
    data,
    model,
    gain_init_calfile=None,
    gain_init_stddev=0.0,
    N_feed_pols=2,
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

    Returns
    -------
    gains_init : array of complex
        Shape (Nants, Nfreqs, 2,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of time intervals.
    Nfreqs : int
        Number of frequency channels.
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
    """

    # Autocorrelations are not currently supported
    data.select(ant_str="cross")
    model.select(ant_str="cross")

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
    else:
        gains_init = initialize_gains_from_calfile(
            gain_init_calfile,
            Nants,
            Nfreqs,
            antenna_list,
            metadata_reference.antenna_names,
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
        visibility_weights[np.where(flag_array)[0]] = 0.0

    return (
        gains_init,
        Nants,
        Nbls,
        Ntimes,
        Nfreqs,
        model_visibilities,
        data_visibilities,
        visibility_weights,
        gains_exp_mat_1,
        gains_exp_mat_2,
    )


def run_calibration_optimization_per_pol(
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
    lambda_val,
    xtol=1e-8,
    verbose=False,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually and the cross-polarization phase is applied from the
    XY and YX visibilities after the fact.

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
        Weight of the phase regularization term; must be positive.
    xtol : float
        Accuracy tolerance for optimizer. Default 1e-8.
    verbose : bool
        Set to True to print optimization outputs.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, Nfreqs, N_feed_pols,).
    """

    gains_fit = np.full(
        (
            Nants,
            Nfreqs,
            N_feed_pols,
        ),
        np.nan,
        dtype=complex,
    )
    for freq_ind in range(Nfreqs):
        for pol_ind in range(N_feed_pols):
            gains_init_flattened = np.stack(
                (
                    np.real(gains_init[:, freq_ind, pol_ind]),
                    np.imag(gains_init[:, freq_ind, pol_ind]),
                ),
                axis=0,
            ).flatten()

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_function_single_pol_wrapper,
                gains_init_flattened,
                args=(
                    Nants,
                    Nbls,
                    model_visibilities[
                        :,
                        :,
                        freq_ind,
                        pol_ind,
                    ],
                    data_visibilities[
                        :,
                        :,
                        freq_ind,
                        pol_ind,
                    ],
                    visibility_weights[
                        :,
                        :,
                        freq_ind,
                        pol_ind,
                    ],
                    gains_exp_mat_1,
                    gains_exp_mat_2,
                    lambda_val,
                ),
                method="Newton-CG",
                jac=jacobian_single_pol_wrapper,
                hess=hessian_single_pol_wrapper,
                options={"disp": verbose, "xtol": xtol},
            )
            end_optimize = time.time()
            print(result.message)
            print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
            sys.stdout.flush()

            gains_fit_single_freq = np.reshape(
                result.x,
                (
                    2,
                    Nants,
                ),
            )
            gains_fit_single_freq = (
                gains_fit_single_freq[0, :] + 1.0j * gains_fit_single_freq[1, :]
            )

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.mean(np.sin(np.angle(gains_fit_single_freq))),
                np.mean(np.cos(np.angle(gains_fit_single_freq))),
            )
            gains_fit_single_freq *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

            gains_fit[:, freq_ind, pol_ind] = gains_fit_single_freq

        # Constrain crosspol phase
        crosspol_phase, gains_fit_new = cost_function_calculations.set_crosspol_phase(
            gains_fit[:, freq_ind, :],
            model_visibilities[:, :, freq_ind, 2:],
            data_visibilities[:, :, freq_ind, 2:],
            visibility_weights[:, :, freq_ind, 2:],
            gains_exp_mat_1,
            gains_exp_mat_2,
            inplace=False,
        )
        gains_fit[:, freq_ind, :] = gains_fit_new

    return gains_fit
