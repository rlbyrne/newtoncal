import numpy as np
import calibration_optimization
import calibration_wrappers
import cost_function_calculations
import calibration_qa
import caldata
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
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
            gains_flattened, caldata_obj, 0, 0
        )

        np.testing.assert_allclose(hess - np.conj(hess.T), 0.0 + 1j * 0.0)

    def test_calibration_single_pol_identical_data_no_flags(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.1, lambda_val=100.0)
        print(f"Gains initial: {caldata_obj.gains}")

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

        caldata_obj.calibration_per_pol(
            xtol=1e-8,
            parallel=False,
        )
        print(f"Gains fit final: {caldata_obj.gains}")

        np.testing.assert_allclose(
            np.abs(caldata_obj.gains),
            np.full(
                (caldata_obj.Nants, caldata_obj.Nfreqs, caldata_obj.N_feed_pols), 1.0
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.angle(caldata_obj.gains),
            np.full(
                (caldata_obj.Nants, caldata_obj.Nfreqs, caldata_obj.N_feed_pols), 0.0
            ),
            atol=1e-6,
        )

    def test_calibration_single_pol_identical_data_with_flags(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = caldata.CalData()
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

        caldata_obj.calibration_per_pol(
            xtol=1e-8,
            parallel=False,
        )

        np.testing.assert_allclose(np.abs(caldata_obj.gains), 1.0)
        np.testing.assert_allclose(np.angle(caldata_obj.gains), 0.0, atol=1e-6)

    def test_antenna_flagging(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = caldata.CalData()
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

        caldata_obj.calibration_per_pol(
            xtol=1e-8,
            parallel=False,
        )

        flag_ant_list = caldata_obj.flag_antennas_from_per_ant_cost(
            flagging_threshold=2.5,
            parallel=False,
            return_antenna_flag_list=True,
        )

        np.testing.assert_equal(flag_ant_list[0][0], perturb_antenna_name)

    def test_per_ant_cost_calc(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()
        use_Nfreqs = 3

        # Create more frequencies
        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
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

        per_ant_cost = calibration_qa.calculate_per_antenna_cost(
            caldata_obj, parallel=False
        )
        per_ant_cost_parallelized = calibration_qa.calculate_per_antenna_cost(
            caldata_obj, parallel=True, max_processes=10
        )

        np.testing.assert_allclose(per_ant_cost, per_ant_cost_parallelized, atol=1e-8)

    def test_crosspol_phase(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.0)

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

        # Set crosspol phase
        crosspol_phase = 0.2
        caldata_obj.gains[:, :, 0] *= np.exp(1j * crosspol_phase / 2)
        caldata_obj.gains[:, :, 1] *= np.exp(-1j * crosspol_phase / 2)

        uvcal = caldata_obj.convert_to_uvcal()
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)

        caldata_obj_new = caldata.CalData()
        caldata_obj_new.load_data(data, model, gain_init_stddev=0.0)

        # Unflag all
        caldata_obj_new.visibility_weights = np.ones(
            (
                caldata_obj_new.Ntimes,
                caldata_obj_new.Nbls,
                caldata_obj_new.Nfreqs,
                4,
            ),
            dtype=float,
        )

        crosspol_phase_new = cost_function_calculations.set_crosspol_phase(
            caldata_obj_new.gains[:, 0, :],
            caldata_obj_new.model_visibilities[:, :, 0, 2:],
            caldata_obj_new.data_visibilities[:, :, 0, 2:],
            caldata_obj_new.visibility_weights[:, :, 0, 2:],
            caldata_obj_new.gains_exp_mat_1,
            caldata_obj_new.gains_exp_mat_2,
        )

        np.testing.assert_allclose(crosspol_phase_new, crosspol_phase, atol=1e-8)

    def test_crosspol_phase_pseudoV(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        model.data_array[:, :, 2] = model.data_array[
            :, :, 3
        ]  # Enforce that PQ and QP visibilities are identical
        data = model.copy()

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model, gain_init_stddev=0.0)

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

        # Set crosspol phase
        crosspol_phase = 0.2
        caldata_obj.gains[:, :, 0] *= np.exp(1j * crosspol_phase / 2)
        caldata_obj.gains[:, :, 1] *= np.exp(-1j * crosspol_phase / 2)

        uvcal = caldata_obj.convert_to_uvcal()
        pyuvdata.utils.uvcalibrate(data, uvcal, inplace=True, time_check=False)

        caldata_obj_new = caldata.CalData()
        caldata_obj_new.load_data(data, model, gain_init_stddev=0.0)

        # Unflag all
        caldata_obj_new.visibility_weights = np.ones(
            (
                caldata_obj_new.Ntimes,
                caldata_obj_new.Nbls,
                caldata_obj_new.Nfreqs,
                4,
            ),
            dtype=float,
        )

        crosspol_phase_new = cost_function_calculations.set_crosspol_phase_pseudoV(
            caldata_obj_new.gains[:, 0, :],
            caldata_obj_new.data_visibilities[:, :, 0, 2:],
            caldata_obj_new.visibility_weights[:, :, 0, 2:],
            caldata_obj_new.gains_exp_mat_1,
            caldata_obj_new.gains_exp_mat_2,
        )

        np.testing.assert_allclose(crosspol_phase_new, crosspol_phase, atol=1e-8)

    def test_abscal_amp_jac(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-8
        amplitude_perturbation = 1.3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        caldata_obj = caldata.CalData()
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

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)
        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all

        for phase_ind in range(2):
            delta_phase_array = np.zeros(2)
            delta_phase_array[phase_ind] = delta_val / 2
            cost1 = cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind]
                + delta_phase_array,
                caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
            )
            cost0 = cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind]
                - delta_phase_array,
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

        caldata_obj = caldata.CalData()
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
        np.testing.assert_allclose(hess_amp_amp_approx, hess_amp_amp, rtol=1e-6)
        np.testing.assert_allclose(hess_amp_phase_approx[0], hess_amp_phasex, rtol=1e-6)
        np.testing.assert_allclose(hess_amp_phase_approx[1], hess_amp_phasey, rtol=1e-6)

    def test_abscal(self):

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = model.copy()

        # Apply abcal offsets
        data.data_array *= (
            1.2
            * np.exp(
                1j * (0.001 * data.uvw_array[:, 0] - 0.002 * data.uvw_array[:, 1])
            )[:, np.newaxis, np.newaxis]
        )

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

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

        caldata_obj.abscal(
            xtol=1e-9,
            maxiter=100,
            verbose=True,
        )
        calibration_wrappers.apply_abscal(
            data,
            caldata_obj.abscal_params,
            caldata_obj.feed_polarization_array,
            inplace=True,
        )
        np.testing.assert_allclose(data.data_array, model.data_array, rtol=1e-3)

    def test_dwabscal_amp_jac(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-8
        amplitude_perturbation = 1.3
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        caldata_obj.model_visibilities[:, :, test_freq_ind, :] *= amplitude_perturbation

        use_amps_1 = np.copy(caldata_obj.abscal_params[0, :, test_pol_ind])
        use_amps_1[test_freq_ind] += delta_val / 2
        cost1 = cost_function_calculations.cost_function_dw_abscal(
            use_amps_1,
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )
        use_amps_0 = np.copy(caldata_obj.abscal_params[0, :, test_pol_ind])
        use_amps_0[test_freq_ind] -= delta_val / 2
        cost0 = cost_function_calculations.cost_function_dw_abscal(
            use_amps_0,
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        grad_approx = (cost1 - cost0) / delta_val
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {amp_jac[test_freq_ind]}")

        np.testing.assert_allclose(grad_approx, amp_jac[test_freq_ind], rtol=1e-8)

    def test_dwabscal_phase_jac(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        delta_val = 1e-6
        amplitude_perturbation = 1.3
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        caldata_obj.model_visibilities[:, :, test_freq_ind, :] *= amplitude_perturbation

        for phase_ind in range(2):
            delta_array = np.zeros((2, caldata_obj.Nfreqs), dtype=float)
            delta_array[phase_ind, test_freq_ind] = delta_val
            cost1 = cost_function_calculations.cost_function_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind] + delta_array / 2,
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )
            cost0 = cost_function_calculations.cost_function_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind] - delta_array / 2,
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )

            amp_jac, phase_jac = cost_function_calculations.jacobian_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind],
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )

            grad_approx = (cost1 - cost0) / delta_val

            np.testing.assert_allclose(
                grad_approx, phase_jac[phase_ind, test_freq_ind], rtol=1e-6
            )

    def test_dwabscal_amp_hess(self, verbose=False):

        test_freq_ind = 1
        test_pol_ind = 0
        delta_val = 1e-6
        amplitude_perturbation = 1.3
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        caldata_obj.model_visibilities[:, :, test_freq_ind, :] *= amplitude_perturbation

        use_amps_1 = np.copy(caldata_obj.abscal_params[0, :, test_pol_ind])
        use_amps_1[test_freq_ind] += delta_val / 2
        amp_jac1, phase_jac1 = cost_function_calculations.jacobian_dw_abscal(
            use_amps_1,
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )
        use_amps_0 = np.copy(caldata_obj.abscal_params[0, :, test_pol_ind])
        use_amps_0[test_freq_ind] -= delta_val / 2
        amp_jac0, phase_jac0 = cost_function_calculations.jacobian_dw_abscal(
            use_amps_0,
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        (
            hess_amp_amp,
            hess_amp_phasex,
            hess_amp_phasey,
            hess_phasex_phasex,
            hess_phasey_phasey,
            hess_phasex_phasey,
        ) = cost_function_calculations.hess_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        amp_grad_approx = (amp_jac1 - amp_jac0) / delta_val
        phase_grad_approx = (phase_jac1 - phase_jac0) / delta_val

        np.testing.assert_allclose(
            amp_grad_approx, hess_amp_amp[:, test_freq_ind], rtol=1e-8
        )
        np.testing.assert_allclose(
            phase_grad_approx[0, :], hess_amp_phasex[:, test_freq_ind], rtol=1e-7
        )
        np.testing.assert_allclose(
            phase_grad_approx[1, :], hess_amp_phasey[:, test_freq_ind], rtol=1e-7
        )

    def test_dwabscal_phase_hess(self, verbose=False):

        test_freq_ind = 1
        test_pol_ind = 0
        delta_val = 1e-6
        amplitude_perturbation = 1.3
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        caldata_obj.model_visibilities[:, :, test_freq_ind, :] *= amplitude_perturbation

        for phase_ind in range(2):
            delta_array = np.zeros((2, caldata_obj.Nfreqs), dtype=float)
            delta_array[phase_ind, test_freq_ind] = delta_val
            amp_jac1, phase_jac1 = cost_function_calculations.jacobian_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind] + delta_array / 2,
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )
            use_amps_0 = np.copy(caldata_obj.abscal_params[0, :, test_pol_ind])
            use_amps_0[test_freq_ind] -= delta_val / 2
            amp_jac0, phase_jac0 = cost_function_calculations.jacobian_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind] - delta_array / 2,
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )

            (
                hess_amp_amp,
                hess_amp_phasex,
                hess_amp_phasey,
                hess_phasex_phasex,
                hess_phasey_phasey,
                hess_phasex_phasey,
            ) = cost_function_calculations.hess_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind],
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )

            amp_grad_approx = (amp_jac1 - amp_jac0) / delta_val
            phase_grad_approx = (phase_jac1 - phase_jac0) / delta_val

            if phase_ind == 0:
                np.testing.assert_allclose(
                    amp_grad_approx, hess_amp_phasex[test_freq_ind, :], rtol=1e-5
                )
                np.testing.assert_allclose(
                    phase_grad_approx[0, :],
                    hess_phasex_phasex[:, test_freq_ind],
                    rtol=1e-5,
                )
                np.testing.assert_allclose(
                    phase_grad_approx[1, :],
                    hess_phasex_phasey[:, test_freq_ind],
                    rtol=1e-5,
                )
            elif phase_ind == 1:
                np.testing.assert_allclose(
                    amp_grad_approx, hess_amp_phasey[test_freq_ind, :], rtol=1e-5
                )
                np.testing.assert_allclose(
                    phase_grad_approx[0, :],
                    hess_phasex_phasey[:, test_freq_ind],
                    rtol=1e-5,
                )
                np.testing.assert_allclose(
                    phase_grad_approx[1, :],
                    hess_phasey_phasey[:, test_freq_ind],
                    rtol=1e-5,
                )

    def test_dwabscal_abscal_agreement(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        dwcal_inv_covariance = np.identity(data.Nfreqs, dtype=complex)
        caldata_obj.dwcal_inv_covariance = np.repeat(
            np.repeat(
                np.repeat(
                    dwcal_inv_covariance[np.newaxis, np.newaxis, :, :, np.newaxis],
                    caldata_obj.Ntimes,
                    axis=0,
                ),
                caldata_obj.Nbls,
                axis=1,
            ),
            caldata_obj.N_vis_pols,
            axis=4,
        )  # Construct identity matrix

        caldata_obj.model_visibilities += np.random.normal(
            scale=0.1,
            size=(
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                caldata_obj.N_vis_pols,
            ),
        ) + 1j * np.random.normal(
            scale=0.1,
            size=(
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                caldata_obj.N_vis_pols,
            ),
        )  # Perturb model

        cost_abscal = 0
        for freq_ind in range(use_Nfreqs):
            cost_abscal += cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, freq_ind, test_pol_ind],
                caldata_obj.model_visibilities[:, :, freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, freq_ind, test_pol_ind],
            )
        cost_dwabscal = cost_function_calculations.cost_function_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )
        np.testing.assert_allclose(cost_abscal, cost_dwabscal, rtol=1e-7)

        amp_jac_abscal, phase_jac_abscal = cost_function_calculations.jacobian_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        amp_jac_dwabscal, phase_jac_dwabscal = (
            cost_function_calculations.jacobian_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind],
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )
        )
        np.testing.assert_allclose(
            amp_jac_abscal, amp_jac_dwabscal[test_freq_ind], rtol=1e-7
        )
        np.testing.assert_allclose(
            phase_jac_abscal, phase_jac_dwabscal[:, test_freq_ind], rtol=1e-7
        )

        (
            hess_amp_amp_abscal,
            hess_amp_phasex_abscal,
            hess_amp_phasey_abscal,
            hess_phasex_phasex_abscal,
            hess_phasey_phasey_abscal,
            hess_phasex_phasey_abscal,
        ) = cost_function_calculations.hess_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        (
            hess_amp_amp_dwabscal,
            hess_amp_phasex_dwabscal,
            hess_amp_phasey_dwabscal,
            hess_phasex_phasex_dwabscal,
            hess_phasey_phasey_dwabscal,
            hess_phasex_phasey_dwabscal,
        ) = cost_function_calculations.hess_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        np.testing.assert_allclose(
            hess_amp_amp_abscal,
            hess_amp_amp_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_amp_phasex_abscal,
            hess_amp_phasex_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_amp_phasey_abscal,
            hess_amp_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasex_phasex_abscal,
            hess_phasex_phasex_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasey_phasey_abscal,
            hess_phasey_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasex_phasey_abscal,
            hess_phasex_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )

    def test_dwabscal_abscal_agreement_compact_dwcal(self, verbose=False):

        test_freq_ind = 0
        test_pol_ind = 0
        use_Nfreqs = 3

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        dwcal_inv_covariance = np.identity(data.Nfreqs, dtype=complex)
        caldata_obj.dwcal_inv_covariance = np.repeat(
            np.repeat(
                np.repeat(
                    dwcal_inv_covariance[np.newaxis, np.newaxis, :, :, np.newaxis],
                    1,
                    axis=0,
                ),
                caldata_obj.Nbls,
                axis=1,
            ),
            1,
            axis=4,
        )  # Construct identity matrix in a more compact form

        caldata_obj.model_visibilities += np.random.normal(
            scale=0.1,
            size=(
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                caldata_obj.N_vis_pols,
            ),
        ) + 1j * np.random.normal(
            scale=0.1,
            size=(
                caldata_obj.Ntimes,
                caldata_obj.Nbls,
                caldata_obj.Nfreqs,
                caldata_obj.N_vis_pols,
            ),
        )  # Perturb model

        cost_abscal = 0
        for freq_ind in range(use_Nfreqs):
            cost_abscal += cost_function_calculations.cost_function_abs_cal(
                caldata_obj.abscal_params[0, freq_ind, test_pol_ind],
                caldata_obj.abscal_params[1:, freq_ind, test_pol_ind],
                caldata_obj.model_visibilities[:, :, freq_ind, test_pol_ind],
                caldata_obj.data_visibilities[:, :, freq_ind, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, freq_ind, test_pol_ind],
            )
        cost_dwabscal = cost_function_calculations.cost_function_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )
        np.testing.assert_allclose(cost_abscal, cost_dwabscal, rtol=1e-7)

        amp_jac_abscal, phase_jac_abscal = cost_function_calculations.jacobian_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        amp_jac_dwabscal, phase_jac_dwabscal = (
            cost_function_calculations.jacobian_dw_abscal(
                caldata_obj.abscal_params[0, :, test_pol_ind],
                caldata_obj.abscal_params[1:, :, test_pol_ind],
                caldata_obj.model_visibilities[:, :, :, test_pol_ind],
                caldata_obj.data_visibilities[:, :, :, test_pol_ind],
                caldata_obj.uv_array,
                caldata_obj.visibility_weights[:, :, :, test_pol_ind],
                caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
            )
        )
        np.testing.assert_allclose(
            amp_jac_abscal, amp_jac_dwabscal[test_freq_ind], rtol=1e-7
        )
        np.testing.assert_allclose(
            phase_jac_abscal, phase_jac_dwabscal[:, test_freq_ind], rtol=1e-7
        )

        (
            hess_amp_amp_abscal,
            hess_amp_phasex_abscal,
            hess_amp_phasey_abscal,
            hess_phasex_phasex_abscal,
            hess_phasey_phasey_abscal,
            hess_phasex_phasey_abscal,
        ) = cost_function_calculations.hess_abs_cal(
            caldata_obj.abscal_params[0, test_freq_ind, test_pol_ind],
            caldata_obj.abscal_params[1:, test_freq_ind, test_pol_ind],
            caldata_obj.model_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.data_visibilities[:, :, test_freq_ind, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, test_freq_ind, test_pol_ind],
        )
        (
            hess_amp_amp_dwabscal,
            hess_amp_phasex_dwabscal,
            hess_amp_phasey_dwabscal,
            hess_phasex_phasex_dwabscal,
            hess_phasey_phasey_dwabscal,
            hess_phasex_phasey_dwabscal,
        ) = cost_function_calculations.hess_dw_abscal(
            caldata_obj.abscal_params[0, :, test_pol_ind],
            caldata_obj.abscal_params[1:, :, test_pol_ind],
            caldata_obj.model_visibilities[:, :, :, test_pol_ind],
            caldata_obj.data_visibilities[:, :, :, test_pol_ind],
            caldata_obj.uv_array,
            caldata_obj.visibility_weights[:, :, :, test_pol_ind],
            caldata_obj.dwcal_inv_covariance[:, :, :, :, test_pol_ind],
        )

        np.testing.assert_allclose(
            hess_amp_amp_abscal,
            hess_amp_amp_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_amp_phasex_abscal,
            hess_amp_phasex_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_amp_phasey_abscal,
            hess_amp_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasex_phasex_abscal,
            hess_phasex_phasex_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasey_phasey_abscal,
            hess_phasey_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            hess_phasex_phasey_abscal,
            hess_phasex_phasey_dwabscal[test_freq_ind, test_freq_ind],
            rtol=1e-7,
        )

    def test_dwabscal_jac_wrapper(self, verbose=False):

        delta_val = 1e-8
        amplitude_perturbation = 1.3
        use_Nfreqs = 5

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")
        data.select(polarizations=-5)
        model.select(polarizations=-5)

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.visibility_weights[:, :, 1, :] = (
            0  # Completely flag one frequency channel
        )

        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        unflagged_freq_inds = np.where(
            np.sum(caldata_obj.visibility_weights, axis=(0, 1, 3)) > 0
        )[0]
        test_parameter_ind = np.random.randint(3 * len(unflagged_freq_inds))
        abscal_params_flattened = caldata_obj.abscal_params[
            :, unflagged_freq_inds, 0
        ].flatten()

        use_params_1 = np.copy(abscal_params_flattened)
        use_params_1[test_parameter_ind] += delta_val / 2
        cost1 = calibration_optimization.cost_dw_abscal_wrapper(
            use_params_1, unflagged_freq_inds, caldata_obj
        )
        use_params_0 = np.copy(abscal_params_flattened)
        use_params_0[test_parameter_ind] -= delta_val / 2
        cost0 = calibration_optimization.cost_dw_abscal_wrapper(
            use_params_0, unflagged_freq_inds, caldata_obj
        )

        jac = calibration_optimization.jacobian_dw_abscal_wrapper(
            abscal_params_flattened, unflagged_freq_inds, caldata_obj
        )

        grad_approx = (cost1 - cost0) / delta_val
        if verbose:
            print(f"Gradient approximation value: {grad_approx}")
            print(f"Jacobian value: {jac[test_parameter_ind]}")

        np.testing.assert_allclose(grad_approx, jac[test_parameter_ind], rtol=1e-6)

    def test_dwabscal_hess_wrapper(self, verbose=False):

        delta_val = 1e-8
        amplitude_perturbation = 1.3
        use_Nfreqs = 5

        model = pyuvdata.UVData()
        model.read(f"{THIS_DIR}/data/test_model_1freq.uvfits")
        data = pyuvdata.UVData()
        data.read(f"{THIS_DIR}/data/test_data_1freq.uvfits")
        data.select(polarizations=-5)
        model.select(polarizations=-5)

        data_copy = data.copy()
        model_copy = model.copy()
        for ind in range(1, use_Nfreqs):
            data_copy.freq_array += 1e6 * ind
            model_copy.freq_array += 1e6 * ind
            data.fast_concat(data_copy, "freq", inplace=True)
            model.fast_concat(model_copy, "freq", inplace=True)

        caldata_obj = caldata.CalData()
        caldata_obj.load_data(data, model)

        caldata_obj.visibility_weights[:, :, :, :] = 1  # Unflag all
        caldata_obj.visibility_weights[:, :, 1, :] = (
            0  # Completely flag one frequency channel
        )

        caldata_obj.dwcal_inv_covariance = np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        ) + 1j * np.random.rand(
            caldata_obj.Ntimes,
            caldata_obj.Nbls,
            caldata_obj.Nfreqs,
            caldata_obj.Nfreqs,
            caldata_obj.N_vis_pols,
        )
        caldata_obj.dwcal_inv_covariance = np.transpose(
            np.matmul(
                np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 2, 3)),
                np.conj(
                    np.transpose(caldata_obj.dwcal_inv_covariance, axes=(0, 1, 4, 3, 2))
                ),
            ),
            axes=(0, 1, 3, 4, 2),
        )  # Enforce that the matrix is Hermitian

        unflagged_freq_inds = np.where(
            np.sum(caldata_obj.visibility_weights, axis=(0, 1, 3)) > 0
        )[0]
        abscal_params_flattened = caldata_obj.abscal_params[
            :, unflagged_freq_inds, 0
        ].flatten()

        hess = calibration_optimization.hessian_dw_abscal_wrapper(
            abscal_params_flattened, unflagged_freq_inds, caldata_obj
        )
        np.testing.assert_allclose(
            hess, hess.T, rtol=1e-8
        )  # Make sure the Hessian is symmetric

        for test_parameter_ind in range(3 * len(unflagged_freq_inds)):
            use_params_1 = np.copy(abscal_params_flattened)
            use_params_1[test_parameter_ind] += delta_val / 2
            jac1 = calibration_optimization.jacobian_dw_abscal_wrapper(
                use_params_1, unflagged_freq_inds, caldata_obj
            )
            use_params_0 = np.copy(abscal_params_flattened)
            use_params_0[test_parameter_ind] -= delta_val / 2
            jac0 = calibration_optimization.jacobian_dw_abscal_wrapper(
                use_params_0, unflagged_freq_inds, caldata_obj
            )

            hess_approx = (jac1 - jac0) / delta_val

            np.testing.assert_allclose(
                hess_approx, hess[test_parameter_ind, :], rtol=1e-5
            )
            np.testing.assert_allclose(
                hess_approx, hess[:, test_parameter_ind], rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
