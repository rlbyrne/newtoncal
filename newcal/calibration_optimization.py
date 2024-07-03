import numpy as np
import sys
import scipy
import scipy.optimize
import time
from newcal import cost_function_calculations


def cost_function_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
    freq_ind,
    vis_pol_ind,
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
    caldata_obj : CalData
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    gains = np.reshape(
        gains_flattened,
        (
            2,
            caldata_obj.Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    cost = cost_function_calculations.cost_function_single_pol(
        gains,
        np.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    return cost


def jacobian_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
    freq_ind,
    vis_pol_ind,
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
    caldata_obj : CalData
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

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
            caldata_obj.Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    jac = cost_function_calculations.jacobian_single_pol(
        gains,
        np.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    jac_flattened = np.stack((np.real(jac), np.imag(jac)), axis=0).flatten()
    return jac_flattened


def hessian_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
    freq_ind,
    vis_pol_ind,
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
    caldata_obj : CalData
    freq_ind : int
        Frequency channel index.
    vis_pol_ind : int
        Index of the visibility polarization.

    Returns
    -------
    hess_flattened : array of float
        Hessian of the cost function, shape (2*Nants, 2*Nants,).
    """

    gains = np.reshape(
        gains_flattened,
        (
            2,
            caldata_obj.Nants,
        ),
    )
    gains = gains[0, :] + 1.0j * gains[1, :]
    (
        hess_real_real,
        hess_real_imag,
        hess_imag_imag,
    ) = cost_function_calculations.hessian_single_pol(
        gains,
        caldata_obj.Nants,
        caldata_obj.Nbls,
        np.reshape(
            caldata_obj.model_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.data_visibilities[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        np.reshape(
            caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind],
            (caldata_obj.Ntimes, caldata_obj.Nbls),
        ),
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    hess_flattened = np.full(
        (2 * caldata_obj.Nants, 2 * caldata_obj.Nants), np.nan, dtype=float
    )
    hess_flattened[0 : caldata_obj.Nants, 0 : caldata_obj.Nants] = hess_real_real
    hess_flattened[caldata_obj.Nants :, 0 : caldata_obj.Nants] = hess_real_imag
    hess_flattened[0 : caldata_obj.Nants, caldata_obj.Nants :] = np.conj(
        hess_real_imag
    ).T
    hess_flattened[caldata_obj.Nants :, caldata_obj.Nants :] = hess_imag_imag
    return hess_flattened


def cost_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function cost_function_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    cost = cost_function_calculations.cost_function_abs_cal(
        abscal_parameters[0],
        abscal_parameters[1:],
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, 0, 0],
    )
    return cost


def jacobian_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function jacobian_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    jac : array of float
        Shape (3,).
    """

    jac = np.zeros((3,), dtype=float)
    amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
        abscal_parameters[0],
        abscal_parameters[1:],
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, 0, 0],
    )
    jac[0] = amp_jac
    jac[1:] = phase_jac
    return jac


def hessian_abscal_wrapper(abscal_parameters, caldata_obj):
    """
    Wrapper for function hess_abs_cal.

    Parameters
    ----------
    abscal_parameters : array of float
        Shape (3,).
    caldata_obj : CalData

    Returns
    -------
    hess : array of float
        Shape (3, 3,).
    """

    hess = np.zeros((3, 3), dtype=float)
    (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    ) = cost_function_calculations.hess_abs_cal(
        abscal_parameters[0],
        abscal_parameters[1:],
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, 0, 0],
    )
    hess[0, 0] = hess_amp_amp
    hess[0, 1] = hess[1, 0] = hess_amp_phasex
    hess[0, 2] = hess[2, 0] = hess_amp_phasey
    hess[1, 1] = hess_phasex_phasex
    hess[2, 2] = hess_phasey_phasey
    hess[1, 2] = hess[2, 1] = hess_phasex_phasey
    return hess


def cost_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function cost_function_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs_unflagged,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    cost : float
        Value of the cost function.
    """

    abscal_parameters = np.full((3, caldata_obj.Nfreqs), np.nan)
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    cost = cost_function_calculations.cost_function_dw_abscal(
        abscal_parameters[0, :],
        abscal_parameters[1:, :],
        caldata_obj.model_visibilities[:, :, :, 0],
        caldata_obj.data_visibilities[:, :, :, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, :, 0],
        caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
    )
    return cost


def jacobian_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function jacobian_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    jac_flattened : array of float
        Flattened array of derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs,).
    """

    abscal_parameters = np.full((3, caldata_obj.Nfreqs), np.nan)
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
        abscal_parameters[0, :],
        abscal_parameters[1:, :],
        caldata_obj.model_visibilities[:, :, :, 0],
        caldata_obj.data_visibilities[:, :, :, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, :, 0],
        caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
    )
    jac_array = np.zeros((3, caldata_obj.Nfreqs), dtype=float)
    jac_array[0, :] = amp_jac
    jac_array[1:, :] = phase_jac
    jac_array = np.take(jac_array, unflagged_freq_inds, axis=1)
    return jac_array.flatten()


def hessian_dw_abscal_wrapper(
    abscal_parameters_flattened, unflagged_freq_inds, caldata_obj
):
    """
    Wrapper for function hess_dw_abscal.

    Parameters
    ----------
    abscal_parameters_flattened : array of float
        Abscal parameters, flattened across the frequency axis. Shape (3 * Nfreqs,).
    unflagged_freq_inds : array of int
        Array of indices of frequency channels that are not fully flagged. Shape (Nfreqs_unflagged,).
    caldata_obj : CalData

    Returns
    -------
    hess : array of float
        Array of second derivatives of the cost function with respect to the abscal
        parameters. Shape (3 * Nfreqs, 3 * Nfreqs,).
    """

    abscal_parameters = np.full((3, caldata_obj.Nfreqs), np.nan)
    abscal_parameters[:, unflagged_freq_inds] = np.reshape(
        abscal_parameters_flattened, (3, len(unflagged_freq_inds))
    )
    (
        hess_amp_amp,
        hess_amp_phasex,
        hess_amp_phasey,
        hess_phasex_phasex,
        hess_phasey_phasey,
        hess_phasex_phasey,
    ) = cost_function_calculations.hess_dw_abscal(
        abscal_parameters[0, :],
        abscal_parameters[1:, :],
        caldata_obj.model_visibilities[:, :, :, 0],
        caldata_obj.data_visibilities[:, :, :, 0],
        caldata_obj.uv_array,
        caldata_obj.visibility_weights[:, :, :, 0],
        caldata_obj.dwcal_inv_covariance[:, :, :, :, 0],
    )
    hess = np.zeros((3, caldata_obj.Nfreqs, 3, caldata_obj.Nfreqs), dtype=float)
    hess[0, :, 0, :] = hess_amp_amp
    hess[0, :, 1, :] = hess[1, :, 0, :] = hess_amp_phasex
    hess[0, :, 2, :] = hess[2, :, 0, :] = hess_amp_phasey
    hess[1, :, 1, :] = hess_phasex_phasex
    hess[2, :, 2, :] = hess_phasey_phasey
    hess[1, :, 2, :] = hess[2, :, 1, :] = hess_phasex_phasey
    hess = np.take(
        np.take(hess, unflagged_freq_inds, axis=1), unflagged_freq_inds, axis=3
    )
    hess = np.reshape(
        hess, (3 * len(unflagged_freq_inds), 3 * len(unflagged_freq_inds))
    )
    return hess


def run_calibration_optimization_per_pol_single_freq(
    caldata_obj,
    xtol,
    maxiter,
    freq_ind=0,
    verbose=True,
    get_crosspol_phase=True,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually. If get_crosspol_phase is set, the cross-
    polarization phase is applied from the XY and YX visibilities after the
    fact.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    freq_ind : int
        Frequency channel to process. Default 0.
    verbose : bool
        Set to True to print optimization outputs. Default True.
    get_crosspol_phase : bool
        Set to True to constrain the cross-polarizaton phase from the XY and YX
        visibilities. Default True.

    Returns
    -------
    gains : array of complex
        Fit gain values. Shape (Nants, 1, N_feed_pols,). Returned only if
        return_gains is True.
    """

    gains_fit = np.ones((caldata_obj.Nants, caldata_obj.N_feed_pols), dtype=complex)
    if np.max(caldata_obj.visibility_weights[:, :, freq_ind, :]) == 0.0:
        print("ERROR: All data flagged.")
        gains_fit[:, :] = np.nan + 1j * np.nan
        return gains_fit

    for feed_pol_ind, feed_pol in enumerate(caldata_obj.feed_polarization_array):
        vis_pol_ind = np.where(caldata_obj.vis_polarization_array == feed_pol)[0]

        if (
            np.max(caldata_obj.visibility_weights[:, :, freq_ind, vis_pol_ind]) == 0.0
        ):  # All flagged
            gains_fit[:, feed_pol_ind] = np.nan + 1j * np.nan
        else:
            gains_init_flattened = np.stack(
                (
                    np.real(caldata_obj.gains[:, freq_ind, feed_pol_ind]),
                    np.imag(caldata_obj.gains[:, freq_ind, feed_pol_ind]),
                ),
                axis=0,
            ).flatten()

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_function_single_pol_wrapper,
                gains_init_flattened,
                args=(caldata_obj, freq_ind, vis_pol_ind),
                method="Newton-CG",
                jac=jacobian_single_pol_wrapper,
                hess=hessian_single_pol_wrapper,
                options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
            )
            end_optimize = time.time()
            if verbose:
                print(result.message)
                print(
                    f"Optimization time: {(end_optimize - start_optimize)/60.} minutes"
                )
            sys.stdout.flush()
            gains_fit_single_pol = np.reshape(result.x, (2, caldata_obj.Nants))
            gains_fit[:, feed_pol_ind] = (
                gains_fit_single_pol[0, :] + 1j * gains_fit_single_pol[1, :]
            )

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit[:, feed_pol_ind]))),
                np.nanmean(np.cos(np.angle(gains_fit[:, feed_pol_ind]))),
            )
            gains_fit[:, feed_pol_ind] *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Constrain crosspol phase
    if (
        get_crosspol_phase
        and caldata_obj.N_feed_pols == 2
        and caldata_obj.N_vis_pols == 4
    ):
        if (
            caldata_obj.feed_polarization_array[0] == -5
            and caldata_obj.feed_polarization_array[1] == -6
        ):
            crosspol_polarizations = [-7, -8]
        elif (
            caldata_obj.feed_polarization_array[0] == -6
            and caldata_obj.feed_polarization_array[1] == -5
        ):
            crosspol_polarizations = [-8, -7]
        crosspol_indices = np.array(
            [
                np.where(caldata_obj.vis_polarization_array == pol)[0][0]
                for pol in crosspol_polarizations
            ]
        )
        crosspol_phase = cost_function_calculations.set_crosspol_phase(
            gains_fit,
            caldata_obj.model_visibilities[:, :, freq_ind, crosspol_indices],
            caldata_obj.data_visibilities[:, :, freq_ind, crosspol_indices],
            caldata_obj.visibility_weights[:, :, freq_ind, crosspol_indices],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
        )

        gains_fit[:, 0] *= np.exp(-1j * crosspol_phase / 2)
        gains_fit[:, 1] *= np.exp(1j * crosspol_phase / 2)

    return gains_fit


def run_abscal_optimization_single_freq(
    caldata_obj,
    xtol,
    maxiter,
    verbose=True,
    return_abscal_params=False,
):
    """
    Run absolute calibration ("abscal").

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    verbose : bool
        Set to True to print optimization outputs. Default True.
    return_abscal_params : bool
        Set to True to return abscal parameter values as an array. Default False.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, 1, N_feed_pols,). Returned only if
        return_abscal_params is True.
    """

    caldata_list = caldata_obj.expand_in_polarization()
    for feed_pol_ind, caldata_per_pol in enumerate(caldata_list):
        # Minimize the cost function
        start_optimize = time.time()
        result = scipy.optimize.minimize(
            cost_abscal_wrapper,
            caldata_per_pol.abscal_params[:, 0, 0],
            args=(caldata_per_pol),
            method="Newton-CG",
            jac=jacobian_abscal_wrapper,
            hess=hessian_abscal_wrapper,
            options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
        )
        caldata_obj.abscal_params[:, 0, feed_pol_ind] = result.x
        end_optimize = time.time()
        if verbose:
            print(result.message)
            print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
        sys.stdout.flush()

    if return_abscal_params:
        return caldata_obj.abscal_params


def run_dw_abscal_optimization(
    caldata_obj,
    xtol,
    maxiter,
    verbose=True,
    return_abscal_params=False,
):
    """
    Run absolute calibration with delay weighting.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    maxiter : int
        Maximum number of iterations for the optimizer.
    verbose : bool
        Set to True to print optimization outputs. Default True.
    return_abscal_params : bool
        Set to True to return abscal parameter values as an array. Default False.

    Returns
    -------
    abscal_params : array of complex
        Fit abscal parameter values. Shape (3, Nfreqs, N_feed_pols,). Returned only if
        return_abscal_params is True.
    """

    caldata_list = caldata_obj.expand_in_polarization()
    for feed_pol_ind, caldata_per_pol in enumerate(caldata_list):
        unflagged_freq_inds = np.where(
            np.sum(caldata_per_pol.visibility_weights, axis=(0, 1, 3)) > 0
        )[0]
        if len(unflagged_freq_inds) == 0:
            print(f"ERROR: Data all flagged.")
            sys.stdout.flush()
            continue
        abscal_params_flattened = caldata_per_pol.abscal_params[
            :, unflagged_freq_inds, 0
        ].flatten()
        # Minimize the cost function
        start_optimize = time.time()
        result = scipy.optimize.minimize(
            cost_dw_abscal_wrapper,
            abscal_params_flattened,
            args=(unflagged_freq_inds, caldata_per_pol),
            method="Newton-CG",
            jac=jacobian_dw_abscal_wrapper,
            hess=hessian_dw_abscal_wrapper,
            options={"disp": verbose, "xtol": xtol, "maxiter": maxiter},
        )
        caldata_obj.abscal_params[:, unflagged_freq_inds, feed_pol_ind] = np.reshape(
            result.x, (3, len(unflagged_freq_inds))
        )
        fully_flagged_freq_inds = np.array(
            [ind for ind in range(caldata_obj.Nfreqs) if ind not in unflagged_freq_inds]
        )
        if verbose:
            print(result.message)
            print(f"Optimization time: {(time.time() - start_optimize)/60.} minutes")
        sys.stdout.flush()

    if return_abscal_params:
        return caldata_obj.abscal_params
