import numpy as np

from numpy.fft import fftn, ifftn
from scipy.interpolate import griddata
from scipy.interpolate import interpn, RegularGridInterpolator

import pyvista as pv
import meshio

from scipy.ndimage import gaussian_filter

from random_fields.correlation_functions import *
# np.random.seed(seed=15)
class Field():

    def __init__(self, model, coordinates):

        self.ndim = coordinates.shape[1]

        self.correlation_model = model
        self.correlation_model.ndim = self.ndim

        self.x_start = None
        self.x_end = None
        self.num_x = None

        self.y_start = None
        self.y_end = None
        self.num_y = None

        self.z_start = None
        self.z_end = None
        self.num_z = None

        self.struct_x = None
        self.struct_y = None
        self.struct_z = None

        self.centroids = None
        self.structured_field = None

        self.coordinates = coordinates

        self.n_points = [300,300,300]

        pass


    def interpolate_to_mesh(self):

        # X1, Y1, Z1 = np.meshgrid(self.struct_x,self.struct_y, self.struct_z, indexing='ij',
        #                       sparse=False)

        self.correlation_model.isometrise_coordinates()

        if self.ndim ==2:
            interpolator = RegularGridInterpolator((self.correlation_model.isotropic_x,
                                                    self.correlation_model.isotropic_y), self.structured_field,
                                                   method='linear')
            self.unstructured_field = interpolator(self.coordinates)
        elif self.ndim ==3:
            interpolator = RegularGridInterpolator((self.correlation_model.isotropic_x,self.correlation_model.isotropic_y, self.correlation_model.isotropic_z), self.structured_field, method='linear')
            self.unstructured_field  = interpolator(self.coordinates)


        # normalize unstructured field
        self.unstructured_field = (self.unstructured_field- self.correlation_model.mu)/np.std(self.unstructured_field)

        self.unstructured_field = self.unstructured_field * self.correlation_model.var + self.correlation_model.mu

        # tmp1 = pv.UnstructuredGrid(self.coordinates)
        # grid = pv.StructuredGrid(X1, Y1, Z1)
        #
        # tmp = grid.interpolate(self.coordinates)
        #
        # self.unstructured_field = interpn((self.struct_x, self.struct_y , self.struct_z), self.structured_field, self.coordinates)
        #
        # # self.unstructured_field = griddata((X1, Y1, Z1), self.structured_field.ravel(), (X,Y,Z),
        # #                                    method="linear")

    def generate_structured_coordinates(self):

        min_x,max_x = self.coordinates[:,0].min(), self.coordinates[:,0].max()
        min_y, max_y = self.coordinates[:, 1].min(), self.coordinates[:, 1].max()


        self.struct_x = np.linspace(min_x,max_x, self.n_points[0])
        self.struct_y = np.linspace(min_y, max_y, self.n_points[1])


        self.correlation_model.x = self.struct_x
        self.correlation_model.y = self.struct_y


        if self.ndim ==3:
            min_z, max_z = self.coordinates[:, 2].min(), self.coordinates[:, 2].max()
            self.struct_z = np.linspace(min_z, max_z, self.n_points[2])
            self.correlation_model.z = self.struct_z

    def generate_field(self):

        self.structured_cor = self.correlation_model.compute_auto_cor_matrix()

        # if self.ndim ==2:
        #     # phi = np.random.rand(self.n_points[0],self.n_points[1])
        #     phi = np.random.rand(self.n_points[0] // 2, self.n_points[1] // 2)
        #     # Make it symmetric to satisfy phi(-k) = -phi(k)
        #     phi_symm = np.concatenate((-phi[::-1, :], phi), axis=0)  # Symmetrize along x-axis
        #     phi_symm = np.concatenate((-phi_symm[:, ::-1], phi_symm), axis=1)  # Symmetrize along y-axis
        # elif self.ndim ==3:
        #     # Generate a random phase array with uniform distribution in [0,1]
        #     phi = np.random.rand(self.n_points[0] // 2, self.n_points[1] // 2, self.n_points[2] // 2)
        #
        #     # Make it symmetric to satisfy phi(-k) = -phi(k)
        #     phi_symm = np.concatenate((-phi[::-1, :, :], phi), axis=0)  # Symmetrize along x-axis
        #     phi_symm = np.concatenate((-phi_symm[:, ::-1, :], phi_symm), axis=1)  # Symmetrize along y-axis
        #     phi_symm = np.concatenate((-phi_symm[:, :, ::-1], phi_symm), axis=2)  # Symmetrize along z-axis
        #     # phi = np.random.rand(self.n_points[0], self.n_points[1], self.n_points[2])
        # else:
        #     phi_symm = 0
        #
        #
        # # Compute the phase term as exp(2j*pi*phi)
        # phase = np.exp(2j * np.pi * phi_symm)

        self.structured_cor = gaussian_filter(self.structured_cor,3)

        F = np.fft.fftn(self.structured_cor)

        # Generate random field
        self.structured_field = np.real(np.fft.ifftn(np.sqrt(F) * np.fft.fftn(np.random.normal(size=(self.n_points[0], self.n_points[1])))))

        # normalize field to maintain correct std
        self.structured_field = self.structured_field/np.std(self.structured_field)

        # rescale structured field
        self.structured_field = self.structured_field*self.correlation_model.var + self.correlation_model.mu

        # self.structured_field  * np.exp(-0.5 * (xx ** 2 + yy ** 2) / sf ** 2)

        # XX,YY = self.correlation_model.compute_meshgrid()

        # self.structured_field = self.structured_field * self.correlation_model.var * np.exp(-((XX/self.correlation_model.rho_x) ** 2 + (YY/self.correlation_model.rho_y) ** 2))


        # Generate the random field by inverse Fourier transform of sqrt(fft(C))*phase
        # self.structured_field = (ifftn_sc(np.sqrt(fftn_sc(self.structured_cor - np.mean(self.structured_cor))) * phase)).real

        # tmp = fftn_sc(self.structured_cor )
        # self.structured_field= (ifftn_sc(np.sqrt(tmp) * phase)).real

        # self.structured_field = (ifftn_sc(np.sqrt(fftn_sc(self.structured_cor )) * phase)).real




if __name__ == '__main__':
    # a = np.linspace(0, 300)

    # from scipy.stats import norm
    # x = np.linspace(-4, 4, 100)
    #
    # p_1 = norm.pdf(x, 0, 1)
    # p_5 = norm.pdf(x, -0.007, 1)
    # # p_5 = norm.pdf(x, 0, 0.97)
    # # p_15 = norm.pdf(x, 0, 0.95)
    #
    # plt.plot(x,p_1)
    # plt.plot(x, p_5)
    # # plt.plot(x, p_15)
    # plt.show()


    model = ExponentialCorrelation(0,1,1,(40,1,10))

    x_coords = np.linspace(-10,10,200)
    y_coords = np.linspace(-10, 10, 200)

    X,Y = np.meshgrid(x_coords, y_coords,indexing='ij')

    X = np.ravel(X)
    Y= np.ravel(Y)

    coordinates = np.array([X,Y]).T
    # coordinates = np.random.rand(19200,2)*20 - 10

    field = Field(model, coordinates)

    field.generate_structured_coordinates()
    field.generate_field()

    # import cProfile
    #
    # cProfile.run('field.interpolate_to_mesh()', 'profile_random_field')
    field.interpolate_to_mesh()

    f = field.unstructured_field

    reshaped_field = np.reshape(f, (200, 200))
    from scipy.stats import norm

    mu, std = norm.fit(f)

    s = np.ones(len(f))*20
    from mpl_toolkits.mplot3d import Axes3D

    from matplotlib.tri import Triangulation

    triangles = Triangulation(coordinates[:,0], coordinates[:,1])
    # triangles2 = Triangulation(field.correlation_model.i[:, 0], coordinates[:, 1])




    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    cont = ax.tricontourf(triangles, f)

    colorbar = plt.colorbar(cont)
    # ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2], c=f)

    plt.show()


    # plt.imshow(field.structured_field[:,:,150].T)
    # tmp = plt.imshow(field.structured_field[:, :].T)
    # colorbar = plt.colorbar(tmp)
    # plt.show()
