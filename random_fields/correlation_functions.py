import numpy as np
from scipy.fft import ifftn as ifftn_sc
from scipy.fft import fftn as fftn_sc

import matplotlib.pyplot as plt


class BaseCorrelation():

    def __init__(self,x,y,z, var, rho, anisotrophy=(1,1,1)):
        self.rho_x = rho * anisotrophy[0]
        self.rho_y = rho * anisotrophy[1]
        self.rho_z = rho * anisotrophy[2]
        self.var = var
        self.x = x
        self.y = y
        self.z = z

    def compute_auto_cor_matrix(self):
        pass

    def compute_meshgrid(self):

        return np.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=True)



class GaussianCorrelation(BaseCorrelation):
    def __init__(self,x,y,z, var, rho, anisotrophy=(1, 1, 1)):
        super(GaussianCorrelation, self).__init__(x,y,z, var, rho, anisotrophy)


    def compute_auto_cor_matrix(self):

        X, Y, Z = self.compute_meshgrid()

        return self.var * np.exp(-((X / self.rho_x) ** 2 + (Y / self.rho_y) ** 2 + (Z / self.rho_z) ** 2))


class ExponentialCorrelation(BaseCorrelation):
    def __init__(self,x,y,z, var, rho, anisotrophy=(1, 1, 1)):
        super(ExponentialCorrelation, self).__init__(x,y,z, var, rho, anisotrophy)


    def compute_auto_cor_matrix(self):

        X, Y, Z = self.compute_meshgrid()

        return self.var * np.exp(-((X / self.rho_x) + (Y / self.rho_y) + (Z / self.rho_z)))