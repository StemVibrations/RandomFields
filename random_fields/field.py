import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.fft import fftn, ifftn

from random_fields.correlation_functions import *

np.random.seed(14)


class Field:

    def __init__(self, model, coordinates, mu, sigma):

        self.coordinates = coordinates
        self.ndim = coordinates.shape[1]

        self.correlation_model = model
        self.correlation_model.ndim = self.ndim

        # mean and std
        self.mu = mu
        self.sigma = sigma

        # structured coordinates
        self.struct_x = None
        self.struct_y = None
        self.struct_z = None

        # coordinates on which the random field is isotropic
        self.isotropic_x = None
        self.isotropic_y = None
        self.isotropic_z = None

        self.isotropic_field = None
        self.unstructured_field = None

        # number of points to generate the structured random field on
        self.n_points = [300, 300, 300]

        pass

    def isometrise_coordinates(self):
        """
        Calculate coordinates on which random field is isotropic

        :return:
        """
        self.isotropic_x = self.struct_x * self.correlation_model.rho_x
        self.isotropic_y = self.struct_y * self.correlation_model.rho_y

        if self.ndim == 3:
            self.isotropic_z = self.struct_z * self.correlation_model.rho_z

    def define_interpolator(self):
        """
        Defines the interpolator from a regular grid
        :return:
        """

        if self.ndim == 2:
            return RegularGridInterpolator((self.isotropic_x, self.isotropic_y), self.isotropic_field, method='linear')
        elif self.ndim == 3:
            return RegularGridInterpolator((self.isotropic_x, self.isotropic_y, self.isotropic_z),
                                           self.isotropic_field, method='linear')

    def interpolate_to_mesh(self):
        """
        Interpolate an isotropic structured random field to an unstructured mesh
        :return:
        """

        # rescale coordinates such that random field on these coordinates is isotropic
        self.isometrise_coordinates()

        # generate interpolator
        interpolator = self.define_interpolator()

        # interpolate isotropic random field to unstructured grid
        self.unstructured_field = interpolator(self.coordinates)

        # normalise unstructured field
        self.unstructured_field = (self.unstructured_field - np.mean(self.unstructured_field)) \
                                  / np.std(self.unstructured_field)

        # rescale unstructured field
        self.unstructured_field = self.unstructured_field * self.sigma + self.mu

    def initialise(self):
        """
        Initialises the field

        :return:
        """
        self.generate_structured_coordinates()

    def generate_structured_coordinates(self):
        """
        Generates a set of structured coordinates based on the unstructured coordinate limits
        :return:
        """

        # find coordinate limits
        min_x, max_x = self.coordinates[:, 0].min(), self.coordinates[:, 0].max()
        min_y, max_y = self.coordinates[:, 1].min(), self.coordinates[:, 1].max()

        # discretise coordinates
        self.struct_x = np.linspace(min_x, max_x, self.n_points[0])
        self.struct_y = np.linspace(min_y, max_y, self.n_points[1])

        # transfer coordinates to correlation model
        self.correlation_model.x = self.struct_x
        self.correlation_model.y = self.struct_y

        if self.ndim == 3:
            min_z, max_z = self.coordinates[:, 2].min(), self.coordinates[:, 2].max()
            self.struct_z = np.linspace(min_z, max_z, self.n_points[2])
            self.correlation_model.z = self.struct_z

    def generate_random_field(self):
        """
        Generates a random field on a unstructured mesh

        :return:
        """
        self.initialise()
        self.generate_structured_isotropic_field()
        self.interpolate_to_mesh()

    def generate_structured_isotropic_field(self):
        """
        Generates a isotropic random field on a structured mesh. Firstly, the auto correlation matrix is generated,
        while not taking into account scale of fluctuation. Then the autocorrelation matrix is multiplied with a
        set of normally distributed samples in the frequency domain. Lastly the field is transformed to the spatial
        domain.

        :return:
        """

        # compute structured isotropic auto correlation field
        structured_cor = self.correlation_model.compute_auto_cor_matrix()

        # transform correlation field and normal distribution samples
        fft_correlation = np.fft.fftn(structured_cor)
        fft_normal_distribution = fftn(np.random.normal(size=(self.n_points[:self.ndim])))

        # Generate random field
        self.isotropic_field = np.real(ifftn(np.sqrt(fft_correlation) * fft_normal_distribution))

        # normalize field to maintain correct std
        self.isotropic_field = self.isotropic_field / np.std(self.isotropic_field)

        # rescale structured field
        self.isotropic_field = self.isotropic_field * self.sigma + self.mu

    def plot_2d_field(self):

        from matplotlib.tri import Triangulation

        # generate triangles
        triangles = Triangulation(self.coordinates[:, 0], coordinates[:, 1])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # generate filled contour plot
        cont = ax.tricontourf(triangles, self.unstructured_field)

        # add color bar
        plt.colorbar(cont)
        plt.show()


if __name__ == '__main__':
    mu = 0
    sigma = 5

    model = GaussianCorrelation(1, (20, 1, 1))

    # generate coordinates field
    x_coords = np.linspace(-10, 10, 200)
    y_coords = np.linspace(-10, 10, 200)

    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    X = np.ravel(X)
    Y = np.ravel(Y)

    coordinates = np.array([X, Y]).T

    # initialise field
    field = Field(model, coordinates, mu, sigma)

    field.generate_random_field()

    field.plot_2d_field()

    # # ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2], c=f)
    #
    # plt.show()
    #
    # # plt.imshow(field.structured_field[:,:,150].T)
    # # tmp = plt.imshow(field.structured_field[:, :].T)
    # # colorbar = plt.colorbar(tmp)
    # # plt.show()
