import numpy as np
import sys
import scipy
import scipy.optimize
import time
import pyuvdata
from newcal import calibration_wrappers
from newcal import cost_function_calculations


def cost_function_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
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
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.visibility_weights[:, :, 0, 0],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    return cost


def jacobian_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
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
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.visibility_weights[:, :, 0, 0],
        caldata_obj.gains_exp_mat_1,
        caldata_obj.gains_exp_mat_2,
        caldata_obj.lambda_val,
    )
    jac_flattened = np.stack((np.real(jac), np.imag(jac)), axis=0).flatten()
    return jac_flattened


def hessian_single_pol_wrapper(
    gains_flattened,
    caldata_obj,
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
        caldata_obj.model_visibilities[:, :, 0, 0],
        caldata_obj.data_visibilities[:, :, 0, 0],
        caldata_obj.visibility_weights[:, :, 0, 0],
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


def run_calibration_optimization_per_pol_single_freq(
    caldata_obj,
    xtol,
    verbose,
):
    """
    Run calibration per polarization. Here the XX and YY visibilities are
    calibrated individually and the cross-polarization phase is applied from the
    XY and YX visibilities after the fact.

    Parameters
    ----------
    caldata_obj : CalData
    xtol : float
        Accuracy tolerance for optimizer.
    verbose : bool
        Set to True to print optimization outputs.

    Returns
    -------
    gains_fit : array of complex
        Fit gain values. Shape (Nants, N_feed_pols,).
    """

    # Expand CalData object into per-pol objects
    for feed_pol_ind, pol in enumerate(caldata_obj.feed_polarization_array):
        sky_pol_ind = np.where(caldata_obj.vis_polarization_array == pol)[0][0]
        caldata_per_pol = calibration_wrappers.CalData()
        caldata_per_pol.gains = caldata_obj.gains[:, :, [feed_pol_ind]]
        caldata_per_pol.Nants = caldata_obj.Nants
        caldata_per_pol.Nbls = caldata_obj.Nbls
        caldata_per_pol.Ntimes = caldata_obj.Ntimes
        caldata_per_pol.Nfreqs = caldata_obj.Nfreqs
        caldata_per_pol.N_feed_pols = 1
        caldata_per_pol.N_vis_pols = 1
        caldata_per_pol.feed_polarization_array = caldata_obj.feed_polarization_array[
            [feed_pol_ind]
        ]
        caldata_per_pol.vis_polarization_array = caldata_obj.vis_polarization_array[
            [sky_pol_ind]
        ]
        caldata_per_pol.model_visibilities = caldata_obj.model_visibilities[
            :, :, :, [sky_pol_ind]
        ]
        caldata_per_pol.data_visibilities = caldata_obj.data_visibilities[
            :, :, :, [sky_pol_ind]
        ]
        caldata_per_pol.visibility_weights = caldata_obj.visibility_weights[
            :, :, :, [sky_pol_ind]
        ]
        caldata_per_pol.gains_exp_mat_1 = caldata_obj.gains_exp_mat_1
        caldata_per_pol.gains_exp_mat_2 = caldata_obj.gains_exp_mat_2
        caldata_per_pol.antenna_names = caldata_obj.antenna_names
        caldata_per_pol.lambda_val = caldata_obj.lambda_val

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
                caldata_per_pol.antenna_names = caldata_per_pol.antenna_names[use_ants]

            gains_init_flattened = np.stack(
                (
                    np.real(caldata_per_pol.gains[:, 0, 0]),
                    np.imag(caldata_per_pol.gains[:, 0, 0]),
                ),
                axis=0,
            ).flatten()

            # Minimize the cost function
            start_optimize = time.time()
            result = scipy.optimize.minimize(
                cost_function_single_pol_wrapper,
                gains_init_flattened,
                args=(caldata_per_pol),
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
            gains_fit = np.reshape(result.x, (2, caldata_per_pol.Nants))
            gains_fit = gains_fit[0, :] + 1j * gains_fit[1, :]

            # Ensure that the phase of the gains is mean-zero
            # This adds should be handled by the phase regularization term, but
            # this step removes any optimizer precision effects.
            avg_angle = np.arctan2(
                np.nanmean(np.sin(np.angle(gains_fit))),
                np.nanmean(np.cos(np.angle(gains_fit))),
            )
            gains_fit *= np.cos(avg_angle) - 1j * np.sin(avg_angle)
            caldata_per_pol.gains = gains_fit[:, np.newaxis, np.newaxis]

        else:  # All flagged
            caldata_per_pol.gains[:, :, :] = np.nan + 1j * np.nan

        if caldata_per_pol.Nants == caldata_obj.Nants:
            caldata_obj.gains[:, :, feed_pol_ind] = caldata_per_pol.gains[:, :, 0]
        else:
            ant_inds = np.array(
                [
                    np.where(caldata_obj.antenna_names == ant_name)[0]
                    for ant_name in caldata_per_pol.antenna_names
                ]
            )
            caldata_obj.gains[ant_inds, :, feed_pol_ind] = caldata_per_pol.gains

    # Constrain crosspol phase
    if caldata_obj.N_feed_pols == 2 and caldata_obj.N_vis_pols == 4:
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
                np.where(caldata_obj.vis_polarization_array == pol)[0]
                for pol in crosspol_polarizations
            ]
        )
        crosspol_phase = cost_function_calculations.set_crosspol_phase(
            caldata_obj.gains[:, 0, :],
            caldata_obj.model_visibilities[:, :, 0, crosspol_indices],
            caldata_obj.data_visibilities[:, :, 0, crosspol_indices],
            caldata_obj.visibility_weights[:, :, 0, crosspol_indices],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
        )

        caldata_obj.gains[:, :, 0] *= np.exp(-1j * crosspol_phase / 2)
        caldata_obj.gains[:, :, 1] *= np.exp(1j * crosspol_phase / 2)
