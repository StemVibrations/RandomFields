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
        - max_conditioning_points (int):
        - random_field_model_name (str):
        - random_field_model (int):
        - n_dim (int):)
        - seed (int):
        - mean (float):
        - variance (float):
        - vertical_scale_fluctuation (list):
        - anisotropy (list):
        - angle (list):
        - v_dim (int):
        - random_field (:class:)
        - z_kriged_field (array)
        - GaussianProcess (:class:)
        - conditioning_points (array):
        - conditioning_values (array):
        - kriging_mean (array):
        - kriging_std = (array):
        - conditioned_random_field (array):


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
        self.GaussianProcess = GPR()
        self.conditioning_points = None
        self.conditioning_values = None
        self.kriging_mean = None
        self.kriging_std = None
        self.conditioned_random_field = None


    def generate(self, nodes: npt.NDArray[np.float64]) -> None:
        """
        Generate random field

        Args:
            - nodes (list): The nodes of the random field
        
        Raises:

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
            - points (array): The contitioning point coordinates
            - values: array-like: The conditioning point values
            - kernel: (:class: sklearn.gaussian_process.kernels.Kernel): The (calibrated) correlation 
                kernel from sklearn (default = None) 
            - noise_level (float or array): normalised variance of the observed data, as a single float 
                or as an array of the same dimensions as `values` (default = 0.0001)

        Raises:

        Returns:
                
        """
        self.noise_level = noise_level

        

        # check the maximum number of conditioning points to keep the computation and memory cost reasonable 
        if points.shape[0] > self.max_conditioning_points:
            raise Exception(f'Too many conditioning points! There are {max(points.shape)} ' + 
                            f'points, while the maximum allowed amount is {self.max_conditioning_points}.' +
                            f'\n Consider increasing `max_conditioning_points` or use fewer conditioning points.' )


        # scale of fluctuation
        scale_fluctuation = np.ones(self.n_dim) * self.vertical_scale_fluctuation

        # apply the anisotropy to the other dimensions
        mask = np.arange(len(scale_fluctuation)) != self.v_dim
        scale_fluctuation[mask] = scale_fluctuation[mask] * self.anisotropy

        #
        # correct the length scales between libraries
        if self.random_field_model_name == 'Gaussian':
            ls_sklearn  = scale_fluctuation * 2/np.pi
            self.KrigingKernel = WhiteKernel(noise_level) + RBF(length_scale = ls_sklearn)    
        elif self.random_field_model_name == 'Exponential':
            ls_sklearn  = scale_fluctuation / np.sqrt(1/2)
            self.KrigingKernel = WhiteKernel(noise_level) + Matern(length_scale = ls_sklearn, nu = 0.5)    
        elif self.random_field_model_name == 'Matern':
            ls_sklearn  = scale_fluctuation / np.sqrt(1/2)
            self.KrigingKernel = WhiteKernel(noise_level) + Matern(length_scale = ls_sklearn)    

        # fit a GP against conditioning points standardised by simulation field statistics: NOT by conditioning point statistics
        self.GaussianProcess = GPR(kernel = self.KrigingKernel)
        self.GaussianProcess.optimizer = None
        self.GaussianProcess.fit(points,(values - self.mean) / np.sqrt(self.variance)  )
        self.conditioning_points = points
        self.conditioning_values = values
        
        self.kriging_mean = None
        self.kriging_std = None
        self.conditioned_random_field = None

    def generate_conditioned(self, nodes: npt.NDArray[np.float64]) -> None:
        """
        Generate conditioned random field

        Args:
            nodes (list): The nodes of the random field
        
        Raises:

        Returns:
            

        """
        # check dimensions of nodes agrees with dimensions of model
        if nodes.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of nodes: {nodes.shape[1]} do not match dimensions of model: {self.n_dim}')
        #
        # create kriged mean field
        z_kriged_field,std_kriged_field  = self.GaussianProcess.predict(nodes,return_std = True)

        std_kriged_field = np.sqrt(std_kriged_field**2 - self.noise_level)

        std_kriged_field *= np.sqrt(self.variance)
        self.kriging_mean = self.mean + np.sqrt(self.variance) * z_kriged_field
        #
        # create single random field at nodes and conditioning points
        nodes_cpoints = np.vstack([nodes,self.conditioning_points])
        self.generate(nodes_cpoints)

        # Split the generated field into the nodal coordinates and the conditioning points
        # Standardize the distribution to marginal ~N(0,1) 
        z_cond_RF_nodes = (self.random_field.field[:nodes.shape[0]] - self.mean) / np.sqrt(self.variance) 
        z_RF_cond_points = (self.random_field.field[nodes.shape[0]:] - self.mean) / np.sqrt(self.variance) 
        
        # add the noise to the conditioning points
        np.random.seed(self.seed)
        z_RF_cond_points += np.sqrt(self.noise_level) * np.random.normal(size = z_RF_cond_points.shape)
        
        #
        # create kriged mean field of random field 
        GP_RF = GPR(kernel = self.GaussianProcess.kernel_)
        GP_RF.optimizer = None
        GP_RF.fit(self.conditioning_points,z_RF_cond_points)
        GP_RF.L_ = self.GaussianProcess.L_
        z_kriged_nodes_rf = GP_RF.predict(nodes)
        #
        # replace kriged mean field (the kriged parts) to create conditioned random field
        z_cond_RF_nodes -= z_kriged_nodes_rf
        z_cond_RF_nodes += z_kriged_field 
     
        #
        # scale conditioned random field
        self.kriging_std = np.sqrt(self.variance) * std_kriged_field
        self.conditioned_random_field = self.mean + np.sqrt(self.variance) * z_cond_RF_nodes