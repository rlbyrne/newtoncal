import numpy as np
import sys
import scipy
import scipy.optimize
import time
import pyuvdata
from scipy import signal


model_path = "data/test_model_1freq.uvfits"
data_path = "data/test_data_1freq.uvfits"


def cost_function_single_pol(
    gains,
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
    Calculate the cost function (chi-squared) value.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
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

    gains_expanded = (
        np.matmul(gains_exp_mat_1, gains) * np.matmul(gains_exp_mat_2, np.conj(gains))
    )[np.newaxis, :]
    res_vec = model_visibilities - gains_expanded * data_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)
    regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
    cost += regularization_term

    return cost


def jacobian_single_pol(
    gains,
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
    Calculate the Jacobian of the chi-squared.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
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
    jac : array of complex
        Jacobian of the chi-squared cost function, shape (Nants,). The real part
        corresponds to derivatives with respect to the real part of the gains;
        the imaginary part corresponds to derivatives with respect to the
        imaginary part of the gains.
    """

    # Convert gains to visibility space
    # Add time axis
    gains_expanded_1 = np.matmul(gains_exp_mat_1, gains)[np.newaxis, :]
    gains_expanded_2 = np.matmul(gains_exp_mat_2, gains)[np.newaxis, :]

    res_vec = gains_expanded_1 * np.conj(gains_expanded_2) * data_visibilities - model_visibilities
    term1 = np.sum(visibility_weights * gains_expanded_2 * np.conj(data_visibilities) * res_vec, axis=0)
    term1 = np.matmul(gains_exp_mat_1.T, term1)
    term2 = np.sum(visibility_weights * gains_expanded_1 * data_visibilities * np.conj(res_vec), axis=0)
    term2 = np.matmul(gains_exp_mat_2.T, term2)

    regularization_term = (
        lambda_val * 1j * np.sum(np.angle(gains)) * gains / np.abs(gains) ** 2.0
    )
    jac = 2 * (term1 + term2 + regularization_term)

    return jac


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


def calibration_setup(
    data,
    model,
    gain_init_calfile=None,
    gain_init_stddev=0.0,
):
    """
    Generate the quantities needed for calibration.

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

    Returns
    -------
    gains_init : array of complex
        Shape (Nants, Nfreqs,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    Ntimes : int
        Number of time intervals.
    Nfreqs : int
        Number of frequency channels.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Npols,).
    data_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Npols,).
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs, Npols,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    """

    Nants = data.Nants_data
    Nbls = data.Nbls
    Ntimes = data.Ntimes
    Nfreqs = data.Nfreqs
    Npols = data.Npols

    # Format visibilities
    data_visibilities = np.zeros(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            Npols,
        ),
        dtype=complex,
    )
    model_visibilities = np.zeros(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            Npols,
        ),
        dtype=complex,
    )
    flag_array = np.zeros((Ntimes, Nbls, Nfreqs, Npols), dtype=bool)
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

    if not np.max(flag_array):  # Check for flags
        apply_flags = False

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
        gains_init = np.ones((Nants, Nfreqs), dtype=complex)
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
            size=(Nants, Nfreqs),
        ) + 1.0j * np.random.normal(
            0.0,
            gain_init_stddev,
            size=(Nants, Nfreqs),
        )

    visibility_weights = np.ones(
        (
            Ntimes,
            Nbls,
            Nfreqs,
            Npols,
        ),
        dtype=float,
    )
    if not np.max(flag_array):  # Apply flagging
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
