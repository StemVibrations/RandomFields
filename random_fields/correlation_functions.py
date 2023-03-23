import numpy as np

class BaseCorrelation():

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        self.rho_x = rho * anisotrophy[0]
        self.rho_y = rho * anisotrophy[1]
        self.rho_z = rho * anisotrophy[2]
        self.x = None
        self.y = None
        self.z = None

        self.scaled_x = None
        self.scaled_y = None
        self.scaled_z = None

        self.ndim = None

    def compute_auto_cor_matrix(self):
        pass

    def compute_meshgrid(self):

        if self.ndim == 2:
            return np.meshgrid(self.x, self.y, indexing='ij', sparse=True)
        elif self.ndim == 3:
            return np.meshgrid(self.x, self.y, self.z, indexing='ij', sparse=True)



class GaussianCorrelation(BaseCorrelation):
    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(GaussianCorrelation, self).__init__(rho, anisotrophy)


    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix
        :return:
        """

        mesh_coords = self.compute_meshgrid()

        r = np.linalg.norm(np.asarray(mesh_coords, dtype=tuple))

        return np.exp(-(r**2))

        # if self.ndim == 2:
        #     return np.exp(-((mesh_coords[0]) ** 2 + (mesh_coords[1]) ** 2))
        # elif self.ndim == 3:
        #     return np.exp(-((mesh_coords[0] ) ** 2 + (mesh_coords[1]) ** 2 + (mesh_coords[2]) ** 2))


class ExponentialCorrelation(BaseCorrelation):
    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(ExponentialCorrelation, self).__init__(rho, anisotrophy)


    def compute_auto_cor_matrix(self):

        mesh_coords= self.compute_meshgrid()

        if self.ndim== 2:
            return np.exp(-((mesh_coords[0]) + (mesh_coords[1])))

        elif self.ndim == 3:
            return np.exp(-((mesh_coords[0]) + (mesh_coords[1]) + (mesh_coords[2] )))