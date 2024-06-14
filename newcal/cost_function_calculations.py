import numpy as np


def cost_function_single_pol(
    gains,
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

    if lambda_val > 0:
        regularization_term = lambda_val * np.sum(np.angle(gains)) ** 2.0
        cost += regularization_term

    return cost


def jacobian_single_pol(
    gains,
    model_visibilities,
    data_visibilities,
    visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
    lambda_val,
):
    """
    Calculate the Jacobian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
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

    res_vec = (
        gains_expanded_1 * np.conj(gains_expanded_2) * data_visibilities
        - model_visibilities
    )
    term1 = np.sum(
        visibility_weights * gains_expanded_2 * np.conj(data_visibilities) * res_vec,
        axis=0,
    )
    term1 = np.matmul(gains_exp_mat_1.T, term1)
    term2 = np.sum(
        visibility_weights * gains_expanded_1 * data_visibilities * np.conj(res_vec),
        axis=0,
    )
    term2 = np.matmul(gains_exp_mat_2.T, term2)

    jac = 2 * (term1 + term2)

    if lambda_val > 0:
        regularization_term = (
            lambda_val * 1j * np.sum(np.angle(gains)) * gains / np.abs(gains) ** 2.0
        )
        jac += 2 * regularization_term

    return jac


def reformat_baselines_to_antenna_matrix(
    bl_array,
    gains_exp_mat_1,
    gains_exp_mat_2,
    Nants,
    Nbls,
):
    """
    Reformat an array indexed in baselines into a matrix with antenna indices.

    Parameters
    ----------
    bl_array : array of float or complex
        Shape (Nbls, ...,).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.

    Returns
    -------
    antenna matrix : array of float or complex
        Shape (Nants, Nants, ...,). Same dtype as bl_array.
    """

    antenna_matrix = np.zeros_like(
        bl_array[0,],
        dtype=bl_array.dtype,
    )
    antenna_matrix = np.repeat(
        np.repeat(antenna_matrix[np.newaxis,], Nants, axis=0)[np.newaxis,],
        Nants,
        axis=0,
    )
    antenna_numbers = np.arange(Nants)
    antenna1_num = np.matmul(gains_exp_mat_1, antenna_numbers)
    antenna2_num = np.matmul(gains_exp_mat_2, antenna_numbers)
    for bl_ind in range(Nbls):
        antenna_matrix[
            antenna1_num[bl_ind],
            antenna2_num[bl_ind],
        ] = bl_array[
            bl_ind,
        ]
    return antenna_matrix


def hessian_single_pol(
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
    Calculate the Hessian of the cost function.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants,).
    Nants : int
        Number of antennas.
    Nbls : int
        Number of baselines.
    model_visibilities : array of complex
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
    hess_real_real : array of float
        Real-real derivative components of the Hessian of the cost function.
        Shape (Nants, Nants,).
    hess_real_imag : array of float
        Real-imaginary derivative components of the Hessian of the cost
        function. Note that the transpose of this array gives the imaginary-real
        derivative components. Shape (Nants, Nants,).
    hess_imag_imag : array of float
        Imaginary-imaginary derivative components of the Hessian of the cost
        function. Shape (Nants, Nants,).
    """

    gains_expanded_1 = np.matmul(gains_exp_mat_1, gains)
    gains_expanded_2 = np.matmul(gains_exp_mat_2, gains)
    data_squared = np.sum(visibility_weights * np.abs(data_visibilities) ** 2.0, axis=0)
    data_times_model = np.sum(
        visibility_weights * model_visibilities * np.conj(data_visibilities), axis=0
    )

    # Calculate the antenna off-diagonal components
    hess_components = np.zeros((Nbls, 4), dtype=float)
    # Real-real Hessian component:
    hess_components[:, 0] = np.real(
        4 * np.real(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )
    # Real-imaginary Hessian component, term 1:
    hess_components[:, 1] = np.real(
        4 * np.real(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        + 2 * np.imag(data_times_model)
    )
    # Real-imaginary Hessian component, term 2:
    hess_components[:, 2] = np.real(
        4 * np.imag(gains_expanded_1) * np.real(gains_expanded_2) * data_squared
        - 2 * np.imag(data_times_model)
    )
    # Imaginary-imaginary Hessian component:
    hess_components[:, 3] = np.real(
        4 * np.imag(gains_expanded_1) * np.imag(gains_expanded_2) * data_squared
        - 2 * np.real(data_times_model)
    )

    hess_components = reformat_baselines_to_antenna_matrix(
        hess_components,
        gains_exp_mat_1,
        gains_exp_mat_2,
        Nants,
        Nbls,
    )
    hess_real_real = hess_components[:, :, 0] + hess_components[:, :, 0].T
    hess_real_imag = hess_components[:, :, 1] + hess_components[:, :, 2].T
    hess_imag_imag = hess_components[:, :, 3] + hess_components[:, :, 3].T

    # Calculate the antenna diagonals
    hess_diag = 2 * (
        np.matmul(gains_exp_mat_1.T, np.abs(gains_expanded_2) ** 2.0 * data_squared)
        + np.matmul(gains_exp_mat_2.T, np.abs(gains_expanded_1) ** 2.0 * data_squared)
    )
    np.fill_diagonal(hess_real_real, hess_diag)
    np.fill_diagonal(hess_imag_imag, hess_diag)
    np.fill_diagonal(hess_real_imag, 0.0)

    if lambda_val > 0:  # Add regularization term
        gains_weighted = gains / np.abs(gains) ** 2.0
        arg_sum = np.sum(np.angle(gains))
        # Antenna off-diagonals
        hess_real_real += (
            2 * lambda_val * np.outer(np.imag(gains_weighted), np.imag(gains_weighted))
        )
        hess_real_imag -= (
            2 * lambda_val * np.outer(np.imag(gains_weighted), np.real(gains_weighted))
        )
        hess_imag_imag += (
            2 * lambda_val * np.outer(np.real(gains_weighted), np.real(gains_weighted))
        )
        # Antenna diagonals
        hess_real_real += np.diag(
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )
        hess_real_imag -= np.diag(
            2
            * lambda_val
            * arg_sum
            * (np.real(gains_weighted) ** 2.0 - np.imag(gains_weighted) ** 2.0)
        )
        hess_imag_imag -= np.diag(
            4 * lambda_val * arg_sum * np.imag(gains_weighted) * np.real(gains_weighted)
        )

    return hess_real_real, hess_real_imag, hess_imag_imag


def set_crosspol_phase(
    gains,
    crosspol_model_visibilities,
    crosspol_data_visibilities,
    crosspol_visibility_weights,
    gains_exp_mat_1,
    gains_exp_mat_2,
):
    """
    Calculate the cross-polarization phase between the P and Q gains. This
    quantity is not constrained in typical per-polarization calibration but is
    required for polarized imaging. See Byrne et al. 2022 for details of the
    calculation.

    Parameters
    ----------
    gains : array of complex
        Shape (Nants, 2,). gains[:, 0] corresponds to the P-polarized gains and
        gains[:, 1] corresponds to the Q-polarized gains.
    crosspol_model_visibilities :  array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized model visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_data_visibilities : array of complex
        Shape (Ntimes, Nbls, 2,). Cross-polarized data visibilities.
        model_visilibities[:, :, 0] corresponds to the PQ-polarized visibilities
        and model_visilibities[:, :, 1] corresponds to the QP-polarized
        visibilities.
    crosspol_visibility_weights : array of float
        Shape (Ntimes, Nbls, 2).
    gains_exp_mat_1 : array of int
        Shape (Nbls, Nants,).
    gains_exp_mat_2 : array of int
        Shape (Nbls, Nants,).

    Returns
    -------
    crosspol_phase : float
        Cross-polarization phase, in radians.
    """

    gains_expanded_1 = np.matmul(gains_exp_mat_1, gains)[np.newaxis, :, :]
    gains_expanded_2 = np.matmul(gains_exp_mat_2, gains)[np.newaxis, :, :]
    term1 = np.nansum(
        crosspol_visibility_weights[:, :, 0]
        * np.conj(crosspol_model_visibilities[:, :, 0])
        * gains_expanded_1[:, :, 0]
        * np.conj(gains_expanded_2[:, :, 1])
        * crosspol_data_visibilities[:, :, 0]
    )
    term2 = np.nansum(
        crosspol_visibility_weights[:, :, 1]
        * crosspol_model_visibilities[:, :, 1]
        * np.conj(gains_expanded_1[:, :, 1])
        * gains_expanded_2[:, :, 0]
        * np.conj(crosspol_data_visibilities[:, :, 1])
    )
    crosspol_phase = np.angle(term1 + term2)

    return crosspol_phase


def cost_function_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the cost function (chi-squared) value for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    res_vec = (amp**2.0 * np.exp(1j * phase_term))[
        np.newaxis, :
    ] * data_visibilities - model_visibilities
    cost = np.sum(visibility_weights * np.abs(res_vec) ** 2)
    return cost


def jacobian_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the Jacobian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).

    Returns
    -------
    amp_jac : float
        Derivative of the cost with respect to the visibility amplitude term.
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2,).
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    amp_jac = (
        4
        * amp
        * np.sum(
            visibility_weights
            * (amp**2.0 * np.abs(data_visibilities) ** 2.0 - np.real(data_prod))
        )
    )
    phase_jac = (
        2
        * amp**2.0
        * np.sum(
            visibility_weights[:, :, np.newaxis]
            * uv_array[np.newaxis, :, :]
            * np.imag(data_prod)[:, :, np.newaxis],
            axis=(0, 1),
        )
    )

    return amp_jac, phase_jac


def hess_abs_cal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
):
    """
    Calculate the Hessian for absolute calibration.

    Parameters
    ----------
    amp : float
        Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls,).


    Returns
    -------
    hess_amp_amp : float
        Second derivative of the cost with respect to the amplitude term.
    hess_amp_phasex : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in x.
    hess_amp_phasey : float
        Second derivative of the cost with respect to the amplitude term and the phase gradient in y.
    hess_phasex_phasex : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasey_phasey : float
        Second derivative of the cost with respect to the phase gradient in x.
    hess_phasex_phasey : float
        Second derivative of the cost with respect to the phase gradient in x and y.
    """

    phase_term = np.sum(phase_grad[np.newaxis, :] * uv_array, axis=1)
    data_prod = (
        np.exp(1j * phase_term)[np.newaxis, :]
        * data_visibilities
        * np.conj(model_visibilities)
    )

    hess_amp_amp = np.sum(
        visibility_weights
        * (
            12.0 * amp**2.0 * np.abs(data_visibilities) ** 2.0
            - 4.0 * np.real(data_prod)
        )
    )

    hess_amp_phasex = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 0] * np.imag(data_prod))
    )
    hess_amp_phasey = (
        4.0
        * amp
        * np.sum(visibility_weights * uv_array[np.newaxis, :, 1] * np.imag(data_prod))
    )

    hess_phasex_phasex = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 0] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasey_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights * uv_array[np.newaxis, :, 1] ** 2.0 * np.real(data_prod)
        )
    )

    hess_phasex_phasey = (
        2.0
        * amp**2.0
        * np.sum(
            visibility_weights
            * uv_array[np.newaxis, :, 0]
            * uv_array[np.newaxis, :, 1]
            * np.real(data_prod)
        )
    )

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )


def cost_function_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the cost function (chi-squared) value for absolute calibration
    with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs)
    cost = np.real(
        np.sum(
            dwcal_inv_covariance
            * np.conj(res_vec[:, :, :, np.newaxis])
            * res_vec[:, :, np.newaxis, :]
        )
    )
    return cost


def jacobian_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the Jacobian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    -------
    amp_jac : array of float
        Derivative of the cost with respect to the visibility amplitude terms. Shape (Nfreqs,).
    phase_jac : array of float
        Derivatives of the cost with respect to the phase gradient terms. Shape (2, Nfreqs,).

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )
    amp_jac = (
        4
        * amp
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    phase_jac = (
        2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    ).T
    return amp_jac, phase_jac


def hess_dw_abscal(
    amp,
    phase_grad,
    model_visibilities,
    data_visibilities,
    uv_array,
    visibility_weights,
    dwcal_inv_covariance,
):
    """
    Calculate the Hessian for absolute calibration with delay weighting.

    Parameters
    ----------
    amp : array of float
        Shape (Nfreqs,). Overall visibility amplitude.
    phase_grad :  array of float
        Shape (2, Nfreqs,). Phase gradient terms, in units of 1/m.
    model_visibilities : array of complex
        Shape (Ntimes, Nbls, Nfreqs,).
    data_visibilities : array of complex
        Relatively calibrated data. Shape (Ntimes, Nbls, Nfreqs,).
    uv_array : array of float
        Shape(Nbls, 2,)
    visibility_weights : array of float
        Shape (Ntimes, Nbls, Nfreqs,).
    dwcal_inv_covariance : array of complex
        Shape (Ntimes, Nbls, Nfreqs, Nfreqs,).

    Returns
    hess_amp_amp : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term.
    hess_amp_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in x.
    hess_amp_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        amplitude term and the phase gradient in y.
    hess_phasex_phasex : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasey_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x.
    hess_phasex_phasey : array of float
        Shape (Nfreqs, Nfreqs,). Second derivative of the cost with respect to the
        phase gradient in x and y.
    -------

    """

    phase_term = np.sum(
        phase_grad[np.newaxis, :, :] * uv_array[:, :, np.newaxis], axis=1
    )  # Shape (Nbls, Nfreqs,)
    res_vec = np.sqrt(visibility_weights) * (
        (amp[np.newaxis, :] ** 2.0 * np.exp(1j * phase_term))[np.newaxis, :, :]
        * data_visibilities
        - model_visibilities
    )  # Shape (Ntimes, Nbls, Nfreqs,)
    derivative_term = (
        np.sqrt(visibility_weights)
        * np.exp(-1j * phase_term)[np.newaxis, :, :]
        * np.conj(data_visibilities)
    )  # Shape (Ntimes, Nbls, Nfreqs,)

    hess_amp_amp_diagonal_term = 4 * np.real(
        np.sum(
            dwcal_inv_covariance
            * derivative_term[:, :, :, np.newaxis]
            * res_vec[:, :, np.newaxis, :],
            axis=(0, 1, 3),
        )
    )
    hess_amp_amp = (
        8
        * amp[:, np.newaxis]
        * amp[np.newaxis, :]
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * derivative_term[:, :, :, np.newaxis]
                * np.conj(derivative_term[:, :, np.newaxis, :]),
                axis=(0, 1),
            )
        )
    ) + np.diag(hess_amp_amp_diagonal_term)

    hess_amp_phase_diagonal_term = (
        4
        * amp[:, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, :, np.newaxis, np.newaxis]
                * res_vec[:, :, np.newaxis, :, np.newaxis],
                axis=(0, 1, 3),
            )
        )
    )
    hess_amp_phase = (
        4
        * amp[:, np.newaxis, np.newaxis] ** 2.0
        * amp[np.newaxis, :, np.newaxis]
        * np.real(
            np.sum(
                dwcal_inv_covariance[:, :, :, :, np.newaxis]
                * (-1j)
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, :]
                * derivative_term[:, :, np.newaxis, :, np.newaxis]
                * np.conj(derivative_term[:, :, :, np.newaxis, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_amp_phasex = hess_amp_phase[:, :, 0] + np.diag(
        hess_amp_phase_diagonal_term[:, 0]
    )
    hess_amp_phasey = hess_amp_phase[:, :, 1] + np.diag(
        hess_amp_phase_diagonal_term[:, 1]
    )

    hess_phasex_phasex = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasey_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasey = (
        2
        * amp[:, np.newaxis] ** 2.0
        * amp[np.newaxis, :] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, np.newaxis, :]
                * np.conj(derivative_term[:, :, :, np.newaxis]),
                axis=(0, 1),
            )
        )
    )
    hess_phasex_phasex_diagonal_term = (
        -2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasey_phasey_diagonal_term = (
        -2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1] ** 2.0
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasey_diagonal_term = (
        -2
        * amp[:, np.newaxis] ** 2.0
        * np.real(
            np.sum(
                dwcal_inv_covariance
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 0]
                * uv_array[np.newaxis, :, np.newaxis, np.newaxis, 1]
                * derivative_term[:, :, :, np.newaxis]
                * res_vec[:, :, np.newaxis, :],
                axis=(0, 1, 3),
            )
        )
    )
    hess_phasex_phasex += hess_phasex_phasex_diagonal_term
    hess_phasey_phasey += hess_phasey_phasey_diagonal_term
    hess_phasex_phasey += hess_phasex_phasey_diagonal_term

    return (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    )
