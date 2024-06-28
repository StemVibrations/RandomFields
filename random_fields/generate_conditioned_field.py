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





        

