import numpy as np

class BaseCorrelation():

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        self.rho_x = rho * anisotrophy[0]
        self.rho_y = rho * anisotrophy[1]
        self.rho_z = rho * anisotrophy[2]

        self.x = None
        self.y = None
        self.z = None

        self.ndim = None

    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix

        :return:
        """
        raise Exception("this is the base function, use the auto cor matrix in the derived class")

    def compute_meshgrid(self):
        """
        Computes the mesh grid of the coordinates
        :return:
        """

        # set mean of coordinates at 0
        new_x = self.x - np.mean(np.array(self.x))
        new_y = self.y - np.mean(np.array(self.y))

        if self.ndim == 2:
            return np.asarray(np.meshgrid(new_x, new_y, indexing='ij', sparse=True), dtype=tuple)
        elif self.ndim == 3:
            new_z = self.z - np.mean(np.array(self.z))
            return np.asarray(np.meshgrid(new_x, new_y, new_z, indexing='ij', sparse=True), dtype=tuple)


class GaussianCorrelation(BaseCorrelation):

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(GaussianCorrelation, self).__init__(rho, anisotrophy)

    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix

        :return:
        """

        mesh_coords = self.compute_meshgrid()

        dist = np.linalg.norm(mesh_coords)

        return np.exp(-(dist**2))



class ExponentialCorrelation(BaseCorrelation):

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(ExponentialCorrelation, self).__init__(rho, anisotrophy)

    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix

        :return:
        """

        mesh_coords = self.compute_meshgrid()
        dist = np.linalg.norm(mesh_coords)

        return np.exp(-dist)


class SinusoidalCorrelation(BaseCorrelation):

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(SinusoidalCorrelation, self).__init__(rho, anisotrophy)

    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix

        :return:
        """

        mesh_coords = self.compute_meshgrid()
        dist = np.linalg.norm(mesh_coords)

        return np.sin(2.2*dist)/(-2.2*dist)

class SecondOrderAutoregressiveCorrelation(BaseCorrelation):

    def __init__(self, rho, anisotrophy=(1, 1, 1)):
        super(SecondOrderAutoregressiveCorrelation, self).__init__(rho, anisotrophy)

    def compute_auto_cor_matrix(self):
        """
        Computes auto correlation matrix

        :return:
        """

        mesh_coords = self.compute_meshgrid()
        dist = np.linalg.norm(mesh_coords)

        return (1+dist) * np.exp(-dist)