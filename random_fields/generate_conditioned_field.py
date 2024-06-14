from enum import Enum
from typing import List
import numpy as np
import gstools as gs

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, \
        ConstantKernel as C, \
        WhiteKernel as W, Matern


class ModelName(Enum):
    """
    Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
    """
    Gaussian = gs.Gaussian
    Exponential = gs.Exponential
    Matern = gs.Matern
    Linear = gs.Linear


class ConditionedRandomFields:
    """
    Generates standard normal conditioned random fields
    """

    def __init__(self, model_name: ModelName, n_dim: int,
                 mean: float, variance: float,
                 v_scale_fluctuation: float, anisotropy: List[float], angle: List[float],
                 seed: int = 14, v_dim: int = 1) -> None:
        """
        Initialise generation of random fields

        Parameters:
        -----------
        model_name: str
            Name of the model to be used. Options are: "Gaussian", "Exponential", "Matern", "Linear"
        n_dim: int
            The dimension of the random field
        mean: float
            The mean of the random field
        variance: float
            The variance of the random field
        v_scale_fluctuation: float
            The vertical scale of fluctuation of the random field
        anisotropy: list
            The anisotropy of the random field (per dimension)
        angle: list
            The angle of the random field (per dimension)
        seed: int
            The seed number for the random number generator
        v_dim: int
            The dimension of the vertical scale of fluctuation
        """
        # initialise model
        if model_name.name not in ModelName.__members__:
            raise ValueError(f'Model name: {model_name} is not supported')

        if n_dim not in [1, 2, 3]:
            raise ValueError('Only 1, 2, 3 dimensions are supported')

        aux_keys = [anisotropy, angle]
        for key in aux_keys:
            if len(key) != n_dim - 1:
                raise ValueError(f'Number of dimensions: {n_dim} does not match number of\
                                 random field properties: {len(variance)}. It should be {n_dim - 1}.')

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
        self.Z_kriged_field = None



    def set_conditioning_points(self,
                                points: np.ndarray,
                                values: np.ndarray, 
                                kernel = None     ) -> None:
        """
        Initiates the conditioning points and inverts the covariance matrix
        
        Parameters:
        -----------
        points: array-like
            The contitioning point coordinates
        values: array-like
            The conditioning point values
        kernel: sklearn kernel
            The (calibrated) correlation kernel from sklearn
        """

        if max(points.shape) > 2000:
            print('too many conditioning points!')

        if kernel == None:
            kernel = W(0.01) + RBF(length_scale=[1.0]*self.n_dim)    

        self.kernel_ = kernel
        self.GaussianProcess = GPR(kernel = kernel)
        self.GaussianProcess.optimizer = None
        self.GaussianProcess.fit(points,values)
        self.conditioning_points = points
        self.conditioning_values = values
        
        self.Z_kriged_field == None


    def generate(self, nodes: np.ndarray) -> None:
        """
        Generate conditioned random field

        Parameters:
        ------------
        nodes: list
            The nodes of the random field
        """
        # check dimensions of nodes agrees with dimensions of model
        if nodes.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of nodes: {nodes.shape[1]} do not match dimensions of model: {self.n_dim}')

        # scale of fluctuation

        LS_sklearn = np.array(self.GaussianProcess.kernel_.get_params()['k2__length_scale'])
        
        #
        # create kriged mean field
        if not isinstance(self.Z_kriged_field, np.ndarray) :
            print('kriging field for first time')
            self.Z_kriged_field = self.GaussianProcess.predict(nodes)


        f_correct = {'Gaussian':np.sqrt(2 / np.pi),
                    'Exponential':np.sqrt(1 / 2),
                    'Matern':np.sqrt(1/ 2 ) }


        if not self.random_field_model_name in f_correct.keys():
            print('!! random field model',self.random_field_model_name,'not implemented... !!')

        #
        # correct the length scales between libraries
        LS_gstools = LS_sklearn * f_correct[self.random_field_model_name]

        model = self.random_field_model(dim=self.n_dim, var=1., len_scale=LS_gstools, angles=self.angle)
        self.random_field = gs.SRF(model, mean=0., seed=self.seed)

        nodes_cpoints = np.vstack([nodes,self.conditioning_points])
        
        #
        # create single random field at nodes and conditioning points
        self.random_field(nodes_cpoints.T)
        Z_crf_nodes = self.random_field.field[:nodes.shape[0]]
        Z_rf_cpoints = self.random_field.field[nodes.shape[0]:]

        #
        # create kriged mean field of random field 
        GPrf = GPR(kernel = self.GaussianProcess.kernel_)
        GPrf.optimizer = None
        GPrf.fit(self.conditioning_points,Z_rf_cpoints)
        GPrf.L_ = self.GaussianProcess.L_
        Z_kriged_nodes_rf = GPrf.predict(nodes)

        #
        # replace kriged mean field (the kriged parts) to create conditioned random field
        Z_crf_nodes += self.Z_kriged_field 
        Z_crf_nodes -= Z_kriged_nodes_rf
     
        #
        # scale conditioned random field
        self.conditioned_random_field = self.mean + np.sqrt(self.variance) * Z_crf_nodes

        

