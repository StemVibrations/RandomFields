
from numpy.fft import fftn, ifftn
from scipy.stats import norm

import pytest

from random_fields.correlation_functions import *
from random_fields.field import Field

from tests.utils import TestUtils

np.random.seed(14)

class TestField():

    @pytest.mark.skip(reason="work in progress")
    def test_scale_of_fluctuation(self):

        mu =0
        sigma=1

        scale_of_fluct_x = 10
        n_points_x = 1000
        n_points_y = 1000

        rho_x=10
        rho_y=1

        model = GaussianCorrelation(1, (scale_of_fluct_x, 1, 1))

        x_coords = np.linspace(-20, 20, n_points_x)
        y_coords = np.linspace(-20, 20, n_points_y)

        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        X_ravel = np.ravel(X)
        Y_ravel = np.ravel(Y)

        coordinates = np.array([X_ravel, Y_ravel]).T
        # coordinates = np.random.rand(19200,2)*20 - 10


        n_fields = 1000
        all_res = []
        for i in range(n_fields):
            np.random.seed(i * 900)

            field_1 = Field(model, coordinates, mu, sigma)
            field_1.generate_random_field()
            f_1 = field_1.unstructured_field

            X_sparse = X[:,0,None]
            Y_sparse = Y[None,0,:]

            dist = np.linalg.norm((X[:,0,None],Y[None,0,:]))

            rho_exact = np.exp(-((X_sparse/rho_x)**2 + (Y_sparse/rho_y)**2))

            fft_correlation = np.fft.fftn(rho_exact)
            fft_normal_distribution = fftn(np.random.normal(size=(n_points_x,n_points_y)))

            new_field = np.real(ifftn(np.sqrt(fft_correlation) * fft_normal_distribution))

            rho_exact = np.exp(-(dist ** 2))

            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax1 = fig.add_subplot(111)



            imshow = ax1.imshow(new_field.T, vmin=-3.1, vmax=3.1)
            plt.colorbar(imshow)
            plt.show()

            field_1.plot_2d_field()

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            #
            # # generate filled contour plot
            # cont = ax.tricontourf(triangles, self.unstructured_field)
            #
            # # add color bar
            # plt.colorbar(cont)

            #
            # from matplotlib.tri import Triangulation
            #
            # # generate triangles
            # triangles = Triangulation(coordinates[:, 0], coordinates[:, 1])
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            #
            # # generate filled contour plot
            # cont = ax.tricontourf(triangles, new_field)
            #
            # # add color bar
            # plt.colorbar(cont)
            # plt.show()



            transposed_reshaped_field = np.reshape(f, (n_points_x, n_points_y)).T

            dir_idx =0

            k = transposed_reshaped_field.shape[dir_idx]
            j= 0

            rho_all = []

            for j in range(k):
                res_cor = 0
                c = 1 / (sigma ** 2 * (k - j))
                sum = 0
                for i in range(1, k-j):

                    v = transposed_reshaped_field[i, dir_idx]
                    v2 = transposed_reshaped_field[i+j, dir_idx]

                    sum += (v-mu) * (v2 - mu)

                rho_j = c * sum
                rho_all.append(rho_j)

            rho_all = np.array(rho_all)
            rho_true = np.exp(-2*(np.abs(x_coords)/20))

            res = np.mean(transposed_reshaped_field,axis=1)
            all_res.append(res)
            # res = TestUtils.moving_average(transposed_reshaped_field[i, :], 100)
            # all_res.append(res)

            opt_res = 0
            for j in range(k):
                tau = coordinates[j,0]
                opt_res += tau*(rho_all[j] - rho_true[j]) * rho_true[j]

            a=1+1



        all_res = np.array(all_res)
        tmp = np.mean(all_res, axis=0)
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

