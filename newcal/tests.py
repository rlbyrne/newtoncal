import numpy as np
import calibration_optimization
import calibration_wrappers
import cost_function_calculations
import calibration_qa
import pyuvdata
import os
import unittest

# Run all tests with pytest tests.py
# Run one test with pytest tests.py::TestStringMethods::test_name

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestStringMethods(unittest.TestCase):
    def test_cost_single_pol_with_identical_data(self):

        test_freq_ind = 0
        test_pol_ind = 0

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model)

        cost = cost_function_calculations.cost_function_single_pol(
            caldata_obj.gains[:, test_freq_ind, 0],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        np.testing.assert_allclose(cost, 0.0)

    def test_jac_single_pol_real_part(self, verbose=False):

        test_ant_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        grad_approx = (cost1 - cost0) / delta_gain
        jac_value = np.real(jac[test_ant_ind])
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {jac_value}")

        np.testing.assert_allclose(grad_approx, jac_value, rtol=1e-5)

    def test_jac_single_pol_imaginary_part(self, verbose=False):

        test_ant_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= 1j * delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += 1j * delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        grad_approx = (cost1 - cost0) / delta_gain
        jac_value = np.imag(jac[test_ant_ind])
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {jac_value}")

        np.testing.assert_allclose(grad_approx, jac_value, rtol=1e-5)

    def test_jac_single_pol_regularization_real_part(self, verbose=False):

        test_ant_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        grad_approx = (cost1 - cost0) / delta_gain
        jac_value = np.real(jac[test_ant_ind])
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {jac_value}")

        np.testing.assert_allclose(grad_approx, jac_value, rtol=1e-5)

    def test_jac_single_pol_regularization_imaginary_part(self, verbose=False):

        test_ant_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= 1j * delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += 1j * delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        grad_approx = (cost1 - cost0) / delta_gain
        jac_value = np.imag(jac[test_ant_ind])
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {jac_value}")

        np.testing.assert_allclose(grad_approx, jac_value, rtol=1e-5)

    def test_hess_single_pol_different_antennas_real(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 20
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian real-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_real[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian real-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_single_pol_different_antennas_imaginary(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 20
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian imaginary-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_2_ind, test_ant_1_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian imaginary-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_imag_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_single_pol_same_antenna_real(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian real-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_real[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian real-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        # Test if approx. consistent with zero
        np.testing.assert_allclose(grad_approx, hess_value, atol=1e-1)

    def test_hess_single_pol_same_antenna_imaginary(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 0  # Don't test regularization
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian imaginary-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_2_ind, test_ant_1_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        # Test if approx. consistent with zero
        np.testing.assert_allclose(grad_approx, hess_value, atol=1e-1)

        # Test Hessian imaginary-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_imag_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_regularization_single_pol_different_antennas_real(
        self, verbose=False
    ):

        test_ant_1_ind = 10
        test_ant_2_ind = 20
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian real-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_real[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian real-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_regularization_single_pol_different_antennas_imaginary(
        self, verbose=False
    ):

        test_ant_1_ind = 10
        test_ant_2_ind = 20
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian imaginary-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_2_ind, test_ant_1_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian imaginary-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_imag_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_regularization_single_pol_same_antenna_real(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian real-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_real[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

        # Test Hessian real-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        # Test if approx. consistent with zero
        np.testing.assert_allclose(grad_approx, hess_value, atol=1e-1)

    def test_hess_regularization_single_pol_same_antenna_imaginary(self, verbose=False):

        test_ant_1_ind = 10
        test_ant_2_ind = 10
        test_freq_ind = 0
        test_pol_ind = 0
        delta_gain = 1e-8
        lambda_val = 1000
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        caldata_obj.visibility_weights = np.zeros_like(
            caldata_obj.visibility_weights
        )  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
            gains_init[:, test_freq_ind],
            caldata_obj.Nants,
            caldata_obj.Nbls,
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.gains_exp_mat_1,
            caldata_obj.gains_exp_mat_2,
            caldata_obj.lambda_val,
        )

        # Test Hessian imaginary-real component
        grad_approx = np.real(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_real_imag[test_ant_2_ind, test_ant_1_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        # Test if approx. consistent with zero
        np.testing.assert_allclose(grad_approx, hess_value, atol=1e-1)

        # Test Hessian imaginary-imaginary component
        grad_approx = np.imag(jac1[test_ant_2_ind] - jac0[test_ant_2_ind]) / delta_gain
        hess_value = hess_imag_imag[test_ant_1_ind, test_ant_2_ind]
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Hessian value: {hess_value}")

        np.testing.assert_allclose(grad_approx, hess_value, rtol=1e-5)

    def test_hess_hermitian(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        lambda_val = 0.01
        gain_stddev = 0.1

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, lambda_val=lambda_val)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(caldata_obj.Nants, caldata_obj.Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag
        gains_flattened = np.stack(
            (
                np.real(gains_init[:, test_freq_ind]),
                np.imag(gains_init[:, test_freq_ind]),
            ),
            axis=0,
        ).flatten()

        hess = calibration_optimization.hessian_single_pol_wrapper(
            gains_flattened, caldata_obj
        )

        np.testing.assert_allclose(hess - np.conj(hess.T), 0.0 + 1j * 0.0)

    def test_calibration_single_pol_identical_data_no_flags(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.1, lambda_val=100.0)

        # Unflag all
        caldata_obj.visibility_weights = np.ones(
            (
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                4,
            ),
            dtype=float,
        )

        calibration_wrappers.calibrate_caldata_per_pol(
            caldata_obj,
            xtol=1e-8,
            parallel=False,
        )

        np.testing.assert_allclose(np.abs(caldata_obj.gains), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.angle(caldata_obj.gains), 0.0, atol=1e-6)

    def test_calibration_single_pol_identical_data_with_flags(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.1, lambda_val=100.0)

        # Unflag all
        caldata_obj.visibility_weights = np.ones(
            (
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                4,
            ),
            dtype=float,
        )
        # Set flags
        caldata_obj.visibility_weights[2, 10, 0, :] = 0.0
        caldata_obj.visibility_weights[1, 20, 0, :] = 0.0

        calibration_wrappers.calibrate_caldata_per_pol(
            caldata_obj,
            xtol=1e-8,
            parallel=False,
        )

        np.testing.assert_allclose(np.abs(caldata_obj.gains), 1.0)
        np.testing.assert_allclose(np.angle(caldata_obj.gains), 0.0, atol=1e-6)

    def test_antenna_flagging(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.1, lambda_val=100.0)

        # Unflag all
        caldata_obj.visibility_weights = np.ones(
            (
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                4,
            ),
            dtype=float,
        )

        perturb_antenna_name = "Tile048"
        antenna_ind = np.where(caldata_obj.antenna_names == perturb_antenna_name)
        baseline_inds = np.where(
            np.logical_or(
                caldata_obj.gains_exp_mat_1[:, antenna_ind],
                caldata_obj.gains_exp_mat_1[:, antenna_ind],
            )
        )[0]

        np.random.seed(0)
        data_perturbation = 1.0j * np.random.normal(
            0.0,
            np.mean(np.abs(caldata_obj.data_visibilities)),
            size=(
                caldata_obj.Ntimes,
                len(baseline_inds),
                caldata_obj.Nfreqs,
                caldata_obj.N_vis_pols,
            ),
        )
        caldata_obj.data_visibilities[:, baseline_inds, :, :] += data_perturbation

        calibration_wrappers.calibrate_caldata_per_pol(
            caldata_obj,
            xtol=1e-8,
            parallel=False,
        )

        flag_ant_list = calibration_qa.get_antenna_flags_from_per_ant_cost(
            caldata_obj,
            flagging_threshold=2.5,
            update_flags=False,
            parallel=False,
        )

        np.testing.assert_equal(flag_ant_list[0][0], perturb_antenna_name)

    def test_abscal_amp_jac(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-8
        amplitude_perturbation = 1.3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.model_visibilities *= amplitude_perturbation

        cost1 = cost_function_calculations.cost_function_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind] + delta_val / 2,
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        cost0 = cost_function_calculations.cost_function_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind] - delta_val / 2,
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )

        amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        print(cost0)

        grad_approx = (cost1 - cost0) / delta_val
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {amp_jac}")

        np.testing.assert_allclose(grad_approx, amp_jac, rtol=1e-8)

    def test_abscal_phase_jac(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-8

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model)
        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all

        for phase_ind in range(2):
            delta_phase_array = np.zeros(2)
            delta_phase_array[phase_ind] = delta_val / 2
            cost1 = cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind] + delta_phase_array,
                caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            )
            cost0 = cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind] - delta_phase_array,
                caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            )

            amp_jac, phase_jac = cost_function_calculations.jacobian_abs_cal(
                caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
                caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            )

            grad_approx = (cost1 - cost0) / delta_val
            if verbose:
                print(f"Gradient approximation value: {grad_approx}")
                print(f"Jacobian value: {phase_jac[phase_ind]}")

            np.testing.assert_allclose(grad_approx, phase_jac[phase_ind], rtol=1e-8)

    def test_abscal_amp_hess(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-8

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = calibration_wrappers.CalData()
        caldata_obj.load_data(data, model)
        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all

        amp_jac1, phase_jac1 = cost_function_calculations.jacobian_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind] + delta_val / 2,
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        amp_jac0, phase_jac0 = cost_function_calculations.jacobian_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind] - delta_val / 2,
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )

        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )

        hess_amp_amp_approx = (amp_jac1 - amp_jac0) / delta_val
        hess_amp_phase_approx = (phase_jac1 - phase_jac0) / delta_val
        np.testing.assert_allclose(hess_amp_amp_approx, hess_amp_amp, rtol=1e-8)
        np.testing.assert_allclose(hess_amp_phase_approx[0], hess_amp_phasex, rtol=1e-8)
        np.testing.assert_allclose(hess_amp_phase_approx[1], hess_amp_phasey, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
