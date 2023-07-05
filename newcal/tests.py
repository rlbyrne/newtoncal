import numpy as np
import calibration
import cost_function_calculations
import pyuvdata
import os
import unittest

# Run all tests with pytest tests.py
# Run one test with pytest tests.py::TestStringMethods::test_name

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestStringMethods(unittest.TestCase):
    def test_cost_single_pol_with_identical_data(self):

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
        ) = calibration.uvdata_calibration_setup(data, model)

        cost = cost_function_calculations.cost_function_single_pol(
            gains_init[:, test_freq_ind, 0],
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= 1j * delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += 1j * delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_ind] -= 1j * delta_gain / 2
        cost0 = cost_function_calculations.cost_function_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_ind] += 1j * delta_gain / 2
        cost1 = cost_function_calculations.cost_function_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        jac = cost_function_calculations.jacobian_single_pol(
            gains_init[:, test_freq_ind],
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        visibility_weights = np.zeros_like(visibility_weights)  # Don't test data

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag

        gains_init0 = np.copy(gains_init[:, test_freq_ind])
        gains_init0[test_ant_1_ind] -= 1j * delta_gain / 2
        jac0 = cost_function_calculations.jacobian_single_pol(
            gains_init0,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        gains_init1 = np.copy(gains_init[:, test_freq_ind])
        gains_init1[test_ant_1_ind] += 1j * delta_gain / 2
        jac1 = cost_function_calculations.jacobian_single_pol(
            gains_init1,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )
        (
            hess_real_real,
            hess_real_imag,
            hess_imag_imag,
        ) = cost_function_calculations.hessian_single_pol(
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
        ) = calibration.uvdata_calibration_setup(data, model)

        np.random.seed(0)
        gains_init_real = np.random.normal(
            1.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        np.random.seed(0)
        gains_init_imag = 1.0j * np.random.normal(
            0.0,
            gain_stddev,
            size=(Nants, Nfreqs),
        )
        gains_init = gains_init_real + gains_init_imag
        gains_flattened = np.stack(
            (
                np.real(gains_init[:, test_freq_ind]),
                np.imag(gains_init[:, test_freq_ind]),
            ),
            axis=0,
        ).flatten()

        hess = calibration.hessian_single_pol_wrapper(
            gains_flattened,
            Nants,
            Nbls,
            model_visibilities[:, :, test_freq_ind, test_pol_ind],
            data_visibilities[:, :, test_freq_ind, test_pol_ind],
            visibility_weights[:, :, test_freq_ind, test_pol_ind],
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
        )

        np.testing.assert_allclose(hess - np.conj(hess.T), 0.0 + 1j * 0.0)

    def test_calibration_single_pol_identical_data_no_flags(self):

        lambda_val = 100.0

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
        ) = calibration.uvdata_calibration_setup(
            data,
            model,
            gain_init_stddev=0.1,
        )

        # Unflag all
        visibility_weights = np.ones(
            (
                Ntimes,
                Nbls,
                Nfreqs,
                4,
            ),
            dtype=float,
        )

        gains_fit = calibration.run_calibration_optimization_per_pol(
            gains_init,
            Nants,
            Nbls,
            Nfreqs,
            2,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
            xtol=1e-8,
        )

        np.testing.assert_allclose(np.abs(gains_fit), 1.0)
        np.testing.assert_allclose(np.angle(gains_fit), 0.0, atol=1e-6)

    def test_calibration_single_pol_identical_data_with_flags(self):

        lambda_val = 100.0

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
        ) = calibration.uvdata_calibration_setup(
            data,
            model,
            gain_init_stddev=0.1,
        )

        # Unflag all
        visibility_weights = np.ones(
            (
                Ntimes,
                Nbls,
                Nfreqs,
                4,
            ),
            dtype=float,
        )
        # Set flags
        visibility_weights[2, 10, 0, :] = 0.0
        visibility_weights[1, 20, 0, :] = 0.0

        gains_fit = calibration.run_calibration_optimization_per_pol(
            gains_init,
            Nants,
            Nbls,
            Nfreqs,
            2,
            model_visibilities,
            data_visibilities,
            visibility_weights,
            gains_exp_mat_1,
            gains_exp_mat_2,
            lambda_val,
            xtol=1e-8,
        )

        np.testing.assert_allclose(np.abs(gains_fit), 1.0)
        np.testing.assert_allclose(np.angle(gains_fit), 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
