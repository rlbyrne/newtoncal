import numpy as np
import sys
import scipy
import scipy.optimize
import time
import pyuvdata
from scipy import signal


model_path = "dwcal/data/model.uvfits"
data_path = "dwcal/data/data.uvfits"


def chi_squared(
    gains,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
):
    """
    Calculate the chi-squared value.

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

    Returns
    -------
    cost : float
        Value of the chi-squared cost function.
    """

    gains_expanded = np.matmul(gains_exp_mat_1, gains) * np.matmul(
        gains_exp_mat_2, np.conj(gains)
    )
    res_vec = model_visibilities - gains_expanded[np.newaxis, :, :] * data_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)

    return cost


def jacobian(
    gains,
    Nants,
    Nbls,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
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

    Returns
    -------
    jac : array of complex
        Jacobian of the chi-squared cost function, shape (Nants,). The real part
        corresponds to derivatives with respect to the real part of the gains;
        the imaginary part corresponds to derivatives with respect to the
        imaginary part of the gains.
    """

    gains_expanded_1 = np.matmul(gains_exp_mat_1, gains)
    gains_expanded_2 = np.matmul(gains_exp_mat_2, gains)

    term1 = np.sum(
        np.matmul(
            gains_exp_mat_1.T,
            visibility_weights
            * (
                gains_expanded_1
                * np.abs(gains_expanded_2) ** 2.0
                * np.abs(data_visibilities) ** 2.0
                - model_visibilities * gains_expanded_2 * np.conj(data_visibilities)
            ),
        ),
        axis=0,
    )
    term2 = np.sum(
        np.matmul(
            gains_exp_mat_2.T,
            visibility_weights
            * (
                gains_expanded_2
                * np.abs(gains_expanded_1) ** 2.0
                * np.abs(data_visibilities) ** 2.0
                - np.conj(model_visibilities) * gains_expanded_1 * data_visibilities
            ),
        ),
        axis=0,
    )
    jac = 2 * (term1 + term2)

    return jac
