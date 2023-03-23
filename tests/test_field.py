import numpy as np

from numpy.fft import fftn, ifftn
from scipy.interpolate import griddata
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.stats import norm


import pytest

from random_fields.correlation_functions import *
from random_fields.field import Field

from tests.utils import TestUtils

np.random.seed(seed=14)

class TestField():


    def test_scale_of_fluctuation(self):

        scale_of_fluct_x = 5
        n_points_x = 200
        n_points_y = 200

        model = GaussianCorrelation(1, 1, (scale_of_fluct_x, 1, 1))

        x_coords = np.linspace(-10, 10, n_points_x)
        y_coords = np.linspace(-10, 10, n_points_y)

        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        X = np.ravel(X)
        Y = np.ravel(Y)

        coordinates = np.array([X, Y]).T
        # coordinates = np.random.rand(19200,2)*20 - 10

        field = Field(model, coordinates)

        field.generate_structured_coordinates()
        field.generate_field()

        # import cProfile
        #
        # cProfile.run('field.interpolate_to_mesh()', 'profile_random_field')
        field.interpolate_to_mesh()

        f = field.unstructured_field

        transposed_reshaped_field = np.reshape(f, (n_points_x, n_points_y)).T

        all_res = []
        for i in range(transposed_reshaped_field.shape[0]):
            res = TestUtils.moving_average(transposed_reshaped_field[i, :], 100)
            all_res.append(res)
        all_res = np.array(all_res)
        a=1+1



    def test_mu_and_sigma(self):
        """
        Checks if the mean and standard deviation of the random field are maintain

        :return:
        """
        scale_of_fluct_x = 5
        n_points_x = 400
        n_points_y = 400

        mu = 10
        sigma = 5
        rho = 1

        # set correlation model
        model = GaussianCorrelation(rho, (scale_of_fluct_x, 1, 1))

        # generate field coordinates
        x_coords = np.linspace(-10, 10, n_points_x)
        y_coords = np.linspace(-10, 10, n_points_y)

        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        X = np.ravel(X)
        Y = np.ravel(Y)

        coordinates = np.array([X, Y]).T

        # generate multiple fields
        n_fields=200
        all_mu_un, all_std_un, all_mu_str, all_std_str = [],[],[],[]
        for i in range(n_fields):
            np.random.seed(i*900)

            field = Field(model, coordinates,mu,sigma)

            field.generate_random_field()

            f_str = field.isotropic_field
            f_un = field.unstructured_field

            # get mu and sigma from the isotropic and unstructured fields
            mu_un, std_un = norm.fit(f_un)
            mu_str, std_str = norm.fit(f_str)

            all_mu_un.append(mu_un)
            all_std_un.append(std_un)

            all_mu_str.append(mu_str)
            all_std_str.append(std_str)

        # assert
        assert np.mean(all_mu_un) == pytest.approx(mu)
        assert np.mean(all_std_un) == pytest.approx(sigma)

        assert np.mean(all_mu_str) == pytest.approx(mu)
        assert np.mean(all_std_str) == pytest.approx(sigma)

