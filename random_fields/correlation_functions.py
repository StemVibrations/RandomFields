import numpy as np
from scipy.fft import ifftn as ifftn_sc
from scipy.fft import fftn as fftn_sc

import matplotlib.pyplot as plt


class BaseCorrelation():

    def __init__(self,mu, var, rho, anisotrophy=(1,1,1)):
        self.rho_x = rho * anisotrophy[0]
        self.rho_y = rho * anisotrophy[1]
        self.rho_z = rho * anisotrophy[2]
        self.var = var
        self.mu = mu
        self.x = None
        self.y = None
        self.z = None

        self.scaled_x = None
        self.scaled_y = None
        self.scaled_z = None

        self.ndim = None

    def compute_auto_cor_matrix(self):
        pass

    def isometrise_coordinates(self):
        self.isotropic_x = self.x * self.rho_x
        self.isotropic_y = self.y * self.rho_y

        if self.ndim == 3:
            self.isotropic_z = self.z * self.rho_z

    def compute_meshgrid(self):

        if self.ndim ==2:
            return np.meshgrid(self.x, self.y, indexing='ij', sparse=True)
        elif self.ndim == 3:
            return np.meshgrid(self.x, self.y, self.x, indexing='ij', sparse=True)



class GaussianCorrelation(BaseCorrelation):
    def __init__(self,mu, var, rho, anisotrophy=(1, 1, 1)):
        super(GaussianCorrelation, self).__init__(mu, var, rho, anisotrophy)


    def compute_auto_cor_matrix(self):

        mesh_coords = self.compute_meshgrid()

        # norm_x_factor = X.max() - X.min()
        # norm_y_factor = Y.max() - Y.min()
        # norm_x = X/ (X.max() - X.min()) -X.min()
        # norm_y = Y / (Y.max() - Y.min()) - Y.min()
        # norm_z = Z / (Z.max() - Z.min()) - Z.min()


        # s = np.sqrt(np.pi)/2

        s=1

        # self.rho_x = 5
        if self.ndim == 2:
            return self.var  * np.exp(-s*((mesh_coords[0]) ** 2 + (mesh_coords[1]) ** 2))
        elif self.ndim == 3:

            return self.var  *np.exp(-s*((mesh_coords[0] ) ** 2 + (mesh_coords[1] ) ** 2 + (mesh_coords[2] ) ** 2))

        # return  self.var *2* np.exp(-((X / (self.rho_x)) ** 2 + (Y / (self.rho_y)) ** 2 + (Z / self.rho_z) ** 2))/200


class ExponentialCorrelation(BaseCorrelation):
    def __init__(self,mu, var, rho, anisotrophy=(1, 1, 1)):
        super(ExponentialCorrelation, self).__init__(mu, var, rho, anisotrophy)


    def compute_auto_cor_matrix(self):

        mesh_coords= self.compute_meshgrid()

        if self.ndim==2:
            return self.var * np.exp(-((mesh_coords[0]) + (mesh_coords[1])))

        elif self.ndim ==3:
            return self.var * np.exp(-((mesh_coords[0]) + (mesh_coords[1]) + (mesh_coords[2] )))