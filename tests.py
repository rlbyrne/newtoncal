import numpy as np
import calibration
import pyuvdata
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_cost_with_identical_data():

    test_ant_ind = 10
    test_freq_ind = 0
    test_pol_ind = 0
    lambda_val = 0.1

    model = pyuvdata.UVData()
    model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
    data = model.copy()

    (
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
    ) = calibration.calibration_setup(data, model)

    cost = calibration.cost_function_single_pol(
        gains_init[:, test_freq_ind],
        Nants,
        Nbls,
        model_visibilities[:, :, test_freq_ind, test_pol_ind],
        data_visibilities[:, :, test_freq_ind, test_pol_ind],
        visibility_weights[:, :, test_freq_ind, test_pol_ind],
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )

    np.testing.assert_allclose(cost, 0.0)


def test_jac_single_pol_real_part():

    test_ant_ind = 10
    test_freq_ind = 0
    test_pol_ind = 0
    delta_gain = 0.0001
    # lambda_val = 0.1
    lambda_val = 0

    model = pyuvdata.UVData()
    model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
    # data = pyuvdata.UVData()
    # data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")
    data = model.copy()

    (
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
    ) = calibration.calibration_setup(data, model)

    gains_init0 = np.copy(gains_init[:, test_freq_ind])
    gains_init0[test_ant_ind] -= delta_gain / 2
    cost0 = calibration.cost_function_single_pol(
        gains_init0,
        Nants,
        Nbls,
        model_visibilities[:, :, test_freq_ind, test_pol_ind],
        data_visibilities[:, :, test_freq_ind, test_pol_ind],
        visibility_weights[:, :, test_freq_ind, test_pol_ind],
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    gains_init1 = np.copy(gains_init[:, test_freq_ind])
    gains_init1[test_ant_ind] += delta_gain / 2
    cost1 = calibration.cost_function_single_pol(
        gains_init1,
        Nants,
        Nbls,
        model_visibilities[:, :, test_freq_ind, test_pol_ind],
        data_visibilities[:, :, test_freq_ind, test_pol_ind],
        visibility_weights[:, :, test_freq_ind, test_pol_ind],
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    jac = calibration.jacobian_single_pol(
        gains_init[:, test_freq_ind],
        Nants,
        Nbls,
        model_visibilities[:, :, test_freq_ind, test_pol_ind],
        data_visibilities[:, :, test_freq_ind, test_pol_ind],
        visibility_weights[:, :, test_freq_ind, test_pol_ind],
        gains_exp_mat_1,
        gains_exp_mat_2,
        lambda_val,
    )
    print((cost1 - cost0) / delta_gain)
    print(jac)
    print(np.real(jac[test_ant_ind]))
