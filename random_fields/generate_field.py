from enum import Enum
from typing import List
import numpy as np
import numpy.typing as npt
import gstools as gs


from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel , \
        WhiteKernel , Matern


class ModelName(Enum):
    """
    Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
    """
    Gaussian = gs.Gaussian
    Exponential = gs.Exponential
    Matern = gs.Matern
    Linear = gs.Linear


class RandomFields():
    """
    Generates (conditioned) random fields

    Inheritance:
        - None

    Attributes:
        - max_conditioning_points (int): maximum number of 
        - random_field_model_name (str): 
        - random_field_model (int):
        - n_dim (int): number of physical dimesions (1,2 or 3) 
        - seed (int): seed for the random number generator
        - mean (float): unconditioned mean of the random field 
        - variance (float): unconditioned variance of the random field
        - vertical_scale_fluctuation (float): 
        - anisotropy (list): ratio between horizontal scales and vertical scale of fluctuation, shape (`self.n_dims` - 1)
        - angle (list): rotation angles of the principal directions of the scales of fluctuation relative to the vertical, shape (`self.n_dims` - 1)
        - v_dim (int): dimesion number corresponding to the vertical scale of fluctuation
        - random_field (:class:gstools.SRF) random field generator 
        - z_kriged_field (array): standard-normal kriged random field
        - gaussian_process (:class:sklearn.gaussian_process.GaussianProcessRegressor)
        - conditioning_points (array): coordinates of the conditioning points, shape (:,`self.n_dims`)
        - conditioning_values (array): values of the conditioning points, shape (:,)
        - kriging_mean (array): mean of the conditioned random field, shape (:,)
        - kriging_std = (array): standard deviation of the conditioned random field, shape (:,)
        - conditioned_random_field (array): conditioned random field, shape (:,)

    """

    def __init__(self, 
                 model_name: ModelName, 
                 n_dim: int,
                 mean: float, 
                 variance: float,
                 v_scale_fluctuation: float, 
                 anisotropy: List[float], 
                 angle: List[float],
                 seed: int = 14, 
                 v_dim: int = 1,
                 max_conditioning_points:int = 2000) -> None:
        """
        Initialise generation of random fields

        Args:
            - model_name (str): Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
            - n_dim (int): The dimension of the random field
            - mean (float): The mean of the random field
            - variance (float): The variance of the random field
            - v_scale_fluctuation (float): The vertical scale of fluctuation of the random field
            - anisotropy (list): The anisotropy of the random field (per dimension)
            - angle (list): The angle of the random field (per dimension)
            - seed (int): The seed number for the random number generator
            - v_dim (int): The dimension of the vertical scale of fluctuation
            - max_conditioning_points (int): Maximum number of points to be used as conditioning points: increase leads to 
                longer computation times and high memory use due to factorisation of dense matrices

        Raises:
            - ValueError: if model_name.name if not a member of class ModelName
            - ValueError: if the number of dimensions is not in [1,2,3]
            - ValueError: if the number of dimensions is nt consistent with the number of random field properties

        Returns:
            - self
            
        """
        # initialise model
        if model_name.name not in ModelName.__members__:
            raise ValueError(f'Model name: {model_name} is not supported')

        if n_dim not in [1, 2, 3]:
            raise ValueError('Only 1, 2, 3 dimensions are supported')

        aux_keys = [anisotropy, angle]
        for key in aux_keys:
            if len(key) != n_dim - 1:
                raise ValueError(f'Number of dimensions: {n_dim} does not match number of ' + 
                                 f'random field properties: {len(key)}. It should be {n_dim - 1}.')


        self.max_conditioning_points = max_conditioning_points
        self.random_field_model_name = model_name.name
        self.random_field_model = model_name.value
        self.n_dim = n_dim
        self.seed = seed
        self.mean = mean
        self.variance = variance
        self.vertical_scale_fluctuation = v_scale_fluctuation
        self.anisotropy = anisotropy
        self.angle = angle
        self.v_dim = v_dim
        self.random_field = None
        self.z_kriged_field = None
        self.gaussian_process = GPR()
        self.conditioning_points = None
        self.conditioning_values = None
        self.kriging_mean = None
        self.kriging_std = None
        self.conditioned_random_field = None


    def generate(self, nodes: npt.NDArray[np.float64]) -> None:
        """
        Generate random field

        Args:
            - nodes (ndarray): The nodes of the random field. shape (:,`self.n_dim`)
        
        Raises:
            - ValueError: if dimensions of `nodes` do not match dimensions of the model

        Returns:

        """
        # check dimensions of nodes agrees with dimensions of model
        if nodes.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of nodes: {nodes.shape[1]} do not match dimensions of model: {self.n_dim}')

        # scale of fluctuation
        scale_fluctuation = np.ones(self.n_dim) * self.vertical_scale_fluctuation

        # apply the anisotropy to the other dimensions
        mask = np.arange(len(scale_fluctuation)) != self.v_dim
        scale_fluctuation[mask] = scale_fluctuation[mask] * self.anisotropy

        model = self.random_field_model(dim = self.n_dim, 
                                        var = self.variance, 
                                        len_scale = scale_fluctuation, 
                                        angles = self.angle)
        self.random_field = gs.SRF(model, 
                                   mean = self.mean, 
                                   seed = self.seed)
        self.random_field(nodes.T)


    def set_conditioning_points(self,
                                points: npt.NDArray[np.float64],
                                values: npt.NDArray[np.float64],
                                noise_level = 0.0001 ) -> None:
        """
        Initiates the conditioning points and inverts the covariance matrix
        
        Args:
            - points (array): The contitioning point coordinates. Dimensions `(:,self.n_dims)`
            - values: array-like: The conditioning point values.
            - kernel: (:class: sklearn.gaussian_process.kernels.Kernel): The (calibrated) correlation 
                kernel from sklearn (default = None).
            - noise_level (float or array): normalised variance of the observed data, as a single float 
                or as an array of the same dimensions as `values` (default = 0.0001).

        Raises:
            - Exception: if more than `self.max_conditioning_points` conditioning points are specified, 
                to prevent excessive computation and memory usage. 
            - ValueError: if dimensions of `points` do not match dimensions of the model

        Returns:
                
        """

        self.noise_level = noise_level
        self.conditioning_points = points
        self.conditioning_values = values

        # check the maximum number of conditioning points to keep the computation and memory cost reasonable 
        if points.shape[0] > self.max_conditioning_points:
            raise Exception(f'Too many conditioning points! There are {points.shape[0]} ' + 
                            f'points, while the maximum allowed amount is {self.max_conditioning_points}.' +
                            f'\n Consider increasing `max_conditioning_points` or use fewer conditioning points.' )

        # check dimensions of conditioning points agrees with dimensions of model
        if points.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of conditioning points: {points.shape[1]} do not match dimensions of model: {self.n_dim}')


        # scale of fluctuation
        scale_fluctuation = np.ones(self.n_dim) * self.vertical_scale_fluctuation

        # apply the anisotropy to the other dimensions
        mask = np.arange(len(scale_fluctuation)) != self.v_dim
        scale_fluctuation[mask] = scale_fluctuation[mask] * self.anisotropy

        #
        # correct the length scales between libraries
        if self.random_field_model_name == 'Gaussian':
            ls_sklearn  = scale_fluctuation * 2/np.pi
            self.kriging_kernel = WhiteKernel(noise_level) + RBF(length_scale = ls_sklearn)    
        elif self.random_field_model_name == 'Exponential':
            ls_sklearn  = scale_fluctuation / np.sqrt(1/2)
            self.kriging_kernel = WhiteKernel(noise_level) + Matern(length_scale = ls_sklearn, nu = 0.5)    
        elif self.random_field_model_name == 'Matern':
            ls_sklearn  = scale_fluctuation / np.sqrt(1/2)
            self.kriging_kernel = WhiteKernel(noise_level) + Matern(length_scale = ls_sklearn)    

        # fit a GP against conditioning points standardised by simulation field statistics: NOT by conditioning point statistics
        self.gaussian_process = GPR(kernel = self.kriging_kernel)
        self.gaussian_process.optimizer = None
        self.gaussian_process.fit(points,(values - self.mean) / np.sqrt(self.variance)  )
        
        # reset variables
        self.kriging_mean = np.array([])
        self.kriging_std = np.array([])
        self.conditioned_random_field = np.array([])

    def generate_conditioned(self, nodes: npt.NDArray[np.float64]) -> None:
        """
        Generate conditioned random field

        Args:
            - nodes (array): The nodes of the random field, shape (:,`self.n_dims`)
        
        Raises:
            - ValueError: if dimensions of nodes do not match the dimensions of the model

        Returns:
            

        """
        # check dimensions of nodes agrees with dimensions of model
        if nodes.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of nodes: {nodes.shape[1]} do not match dimensions of model: {self.n_dim}')
        #
        # create kriged mean field
        z_kriged_field,std_kriged_field  = self.gaussian_process.predict(nodes,return_std = True)

        std_kriged_field = np.sqrt(std_kriged_field**2 - self.noise_level)

        std_kriged_field *= np.sqrt(self.variance)
        self.kriging_mean = self.mean + np.sqrt(self.variance) * z_kriged_field
        #
        # create single random field at nodes and conditioning points
        self.generate(np.vstack([nodes,self.conditioning_points]))

        # Split the generated field into the nodal coordinates and the conditioning points
        # Standardize the distribution to marginal ~N(0,1) 
        z_cond_rf_nodes = (self.random_field.field[:nodes.shape[0]] - self.mean) / np.sqrt(self.variance) 
        z_rf_cond_points = (self.random_field.field[nodes.shape[0]:] - self.mean) / np.sqrt(self.variance) 
        
        # add the noise to the conditioning points
        np.random.seed(self.seed)
        z_rf_cond_points += np.sqrt(self.noise_level) * np.random.normal(size = z_rf_cond_points.shape)
        
        #
        # create kriged mean field of random field 
        gp_rf = GPR(kernel = self.gaussian_process.kernel_)
        gp_rf.optimizer = None
        gp_rf.fit(self.conditioning_points,z_rf_cond_points)
        gp_rf.L_ = self.gaussian_process.L_
        #
        # replace kriged mean field (the kriged parts) to create conditioned random field
        z_cond_rf_nodes -= gp_rf.predict(nodes)
        z_cond_rf_nodes += z_kriged_field 
     
        #
        # scale conditioned random field
        self.kriging_std = np.sqrt(self.variance) * std_kriged_field
        self.conditioned_random_field = self.mean + np.sqrt(self.variance) * z_cond_rf_nodes