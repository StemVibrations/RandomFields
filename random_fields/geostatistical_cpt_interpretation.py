#%%

from typing import Optional

import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.stats as st

from generate_field import RandomFields, ModelName

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum

from geolib_plus.gef_cpt import GefCpt
from geolib_plus.robertson_cpt_interpretation import RobertsonCptInterpretation
from geolib_plus.robertson_cpt_interpretation import UnitWeightMethod
from geolib_plus.robertson_cpt_interpretation import OCRMethod
from geolib_plus.robertson_cpt_interpretation import ShearWaveVelocityMethod


class MarginalTransformation():
    """Class for the transformation model between the distributions of the pysical values and the standard-normal distribution  
    """
    def __init__(self,
                 data: npt.NDArray[np.float64],
                 min: Optional[float] = None,
                 max: Optional[float] = None,
                 lognormal: Optional[bool] = False):
        """
        Initiate class

        Args:
            - data (array-like): sampling data to construct transformation 
            - min (float, optional): _description_. Defaults to None.
            - max (float, optional): _description_. Defaults to None.
            - lognormal (bool, optional): _description_. Defaults to False.
        """        

        self.data = np.sort(data)
        self.n = len(data)
        self.lognormal = lognormal

        if type(min) == type(None):
            self.min = self.data.min() - (self.data.max() - self.data.min())/100
            self.min_ln = np.log(self.data.min()) - (np.log(self.data.max()) - np.log(self.data.min()))/100
        else:
            self.min = min
            self.min_ln = np.log(min)
        if type(max) == type(None):
            self.max = self.data.max() + (self.data.max() - self.data.min())/100
            self.max_ln = np.log(self.data.max()) + (np.log(self.data.max()) - np.log(self.data.min()))/100
        else:
            self.max = max
            self.max_ln = np.log(max)

        self.z = st.norm.ppf(np.linspace(.5/self.n,1-.5/self.n,self.n))

        x_points = np.hstack([self.min,self.data,self.max])
        x_points_ln = np.hstack([self.min_ln,np.log(self.data),self.max_ln])
        z_points = np.hstack([-8.,self.z,8.])

        self._z = np.linspace(-8,8,101)
        self._data = np.interp(self._z,z_points,x_points)
        self._log_data = np.interp(self._z,z_points,x_points_ln)

    def plot(self,
             title: str = "Marginal distribution transformation ",
             x_label: str = '$Z\sim N(0,1)$',
             y_label: str = '$X $',
             figsize: tuple = (6,5)):
        """Plots the transformation model for the marginal distribution. 

        Args:
            title (str, optional): plot title. Defaults to "Marginal distribution transformation ".
            x_label (str, optional): horizontal axis label. Defaults to '\sim N(0,1)$'.
            y_label (str, optional): vertical axis label. Defaults to ' $'.
            figsize (tuple, optional): figure size. Defaults to (6,5).
        """
        plt.figure(figsize = figsize)
        plt.plot(self.z,self.data,'.',color = 'gray',alpha = 0.5,label = 'data')
        plt.plot(self._z,self._d,'-',color = 'r',label = 'transformation model')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.title(title)
        plt.grid()
        plt.show()

    def XtoZ(self,x: npt.NDArray[np.float64]):
        """XtoZ transforms data from the distribution of $$X$$ to the standard normal 
           equivalent distribution $$Z\sim N(0,1)$$

        Args:
            x (npt.NDArray[np.float64]): values $$x \in X$$

        Returns:
            z (npt.NDArray[np.float64]): standard-normal equivalent to `x`
        """
        if self.lognormal:
            z = np.interp(np.log(x),self._log_data,self._z)
        else:
            z = np.interp(x,self._data,self._z)
        return z

    def ZtoX(self,Z):
        """ZtoX transforms from a standard-normal values `z` to the corresponding values in `x`.

        Args:
            z (npt.NDArray[np.float64]): standard normal values $z \in Z\sim N(0,1)$

        Returns:
            x (npt.NDArray[np.float64]): standard-normal equivalent to `x`
        """
        if self.lognormal:
            x = np.exp(np.interp(Z,self._z,self._log_data))
        else:
            x = np.interp(Z,self._z,self._data)
        return x

class CPT_data():
    """CPT_data _summary_
    
    
    """    

    def __init__(self,cpt_directory = './'):
        """__init__ _summary_

        Args:
            cpt_directory (str, optional): _description_. Defaults to './'.
        """    
        self.gravity = 9.81    
        self.cpt_list = []
        self.cpt_directory = cpt_directory
        self.cpt_filepaths = []

        self.interpreter = RobertsonCptInterpretation()
        self.interpreter.unitweightmethod = UnitWeightMethod.LENGKEEK
        self.interpreter.shearwavevelocitymethod = ShearWaveVelocityMethod.ZANG
        self.interpreter.ocrmethod = OCRMethod.MAYNE

    def read_cpt_data(self,addition = False):
        """read_cpt_data Reads in and pre-processes the CPT data

        Args:
            addition (bool, optional): If True, reads in the CPT data in addition to the existing data. Defaults to False.

        Raises:
            Warning: if .xml-files are encountered in directory
            
        """        
        if not addition:
            self.cpt_list = []
            self.cpt_filepaths = []


        for gef_name in os.listdir(self.cpt_directory):
            if gef_name.endswith('.gef'): 
                print(gef_name)

                cpt = GefCpt()
                cpt.read(self.cpt_directory + '/' + gef_name)
                cpt.pre_process_data()

                self.cpt_list.append(cpt)
                self.cpt_filepaths.append(self.cpt_directory + '/' + gef_name)
                
            elif gef_name.endswith('.xml'):
                try:

                    cpt = GefCpt()
                    cpt.read(self.cpt_directory + '/' + gef_name)
                    cpt.pre_process_data()

                    self.cpt_list.append(cpt)
                    self.cpt_filepaths.append(self.cpt_directory + '/' + gef_name)
                    print(gef_name)
                except:
                    raise Warning('Ignoring file {gef_name}. xml-files for CPT data may not be supported by geolib_plus?') 



    def interpret_cpt_data(self,interpreter = None,v_dim = 1):# = RobertsonCptInterpretation):
        """interpret_cpt_data interprets the CPT data based on the provided interpreter settings

        Args:
            interpreter (:class:`geolib_plus.robertson_cpt_interpretation.RobertsonCptInterpretation`, optional): Interpretation methods. Defaults to RobertsonCptInterpretation.
            
        """

        if interpreter:
            self.interpreter = interpreter

        self.data_coords = np.array([[]]*3,dtype = np.float64).T
        self.cpt_locations = np.array([[]]*2,dtype = np.float64).T
        self.rho = np.array([])
        self.vs = np.array([])
        
        for cpt in self.cpt_list:
            cpt.interpret_cpt(self.interpreter)

            n_readings = len(cpt.depth_to_reference)
            coords = np.array([[cpt.coordinates[0],0,cpt.coordinates[1]]]*n_readings)
            coords[:,v_dim] = cpt.depth_to_reference

            self.cpt_locations = np.vstack([self.cpt_locations, cpt.coordinates])
            self.data_coords = np.vstack([self.data_coords,coords])
            self.rho = np.hstack([self.rho,cpt.rho])
            self.vs = np.hstack([self.vs,cpt.vs])

        self.g_0 = self.rho/self.gravity*self.vs**2

        self.save_conditioning_points_to_vtk(variable_names = ['vs','rho'])

    def save_conditioning_points_to_vtk(self,
                filename = 'conditioning_data.vtk',
                variable_names ):
        """Saves the conditioning data to a .vtk file for visualisation.

        Args:
            - filename (str, optional): . Defaults to 'conditioning_data.vtk'.
            - variable_names (list[str]): list lf variables to save. variables should be attributes.

        Raises:
            ValueError: if type and dimensions of `coordinates` not correct 
            ValueError: if variables in `variable_names` are not an attribute
            ValueError: if variable dimensions are not correct. 
        """
        coordinates = self.data_coords

        if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 3:
            raise ValueError("Coordinates should be a numpy array with shape (N, 3).")

        n_points = len(coordinates) 
        n_values = len(variable_names)

        scalars = []
        for varname in variable_names:
            if hasattr(self,varname):
                scalar = self.__getattribute__(varname)
                if len(scalar) == n_points and isinstance(scalar,np.ndarray):
                    scalars.append(scalar)
                else:
                    raise ValueError(f"`{varname}` should be a numpy array of length {n_points}.")
            else:
                raise ValueError(f"{varname} is not an instance of `conditioning_data`")
       
        if not isinstance(scalars, list) or len(scalars) != len(variable_names) or len(scalars[0]) != coordinates.shape[0]:
            raise ValueError("Scalars should be a numpy array with shape (N, 3), matching the number of coordinates.")

        with open(filename,'w') as file:
            file.write('# vtk DataFile Version 5.1\n')
            file.write('vtk output\n')
            file.write('ASCII\n')
            file.write('DATASET POLYDATA\n')
            file.write(f'POINTS {n_points} float\n')
            for point in coordinates:
                file.write('%10.2f %10.2f %10.2f\n'%(point[0],point[1],point[2]))
            file.write(f'POINT_DATA {n_points}\n')
            file.write(f'FIELD FieldData {n_values}\n')

            for z,name in zip(scalars,variable_names):
                file.write(f'{name} 1 {n_points} float\n')
                for x in z:
                    file.write('%.g '%(x))
                file.write('\n')

        print(f"File saved as {filename}")


    def data_coordinate_change(self,
                               x_ref:float = 0.,
                               y_ref:float = 0.,
                               orientation_x_axis:float = 0.,
                               based_on_midpoint = False):
        """data_coordinate_change Coordinate system transformation, relative to a reference point (x_ref,y_ref)

        Args:
            - x_ref (float, optional): _description_. Defaults to 0..
            - y_ref (float, optional): _description_. Defaults to 0..
            - orientation_x_axis (float, optional): orienation of the model in degrees, clock-wise from North: r_ref = 0 results 
                    in a 90 degrees clock-wise rotation in the x-y plane . Defaults to 0..
            - based_on_midpoint (bool, optional): If True, translates the domain relative to the midpoint of the CPT locations 
                    `self.cpt_locations`. Defaults to False.
        """        
        
        #
        # translate coordinates
        if based_on_midpoint:
            x_ref = (self.cpt_locations[:,0].min() + self.cpt_locations[:,0].max())/2.
            y_ref = (self.cpt_locations[:,1].min() + self.cpt_locations[:,1].max())/2.

        self.data_coords[:,0] -= x_ref
        self.data_coords[:,2] -= y_ref
        self.cpt_locations[:,0] -= x_ref
        self.cpt_locations[:,1] -= y_ref

        # rotate counterclock-wise back to N000  minus 90 degrees:
        theta_rad = orientation_x_axis/180*np.pi  - np.pi/2
        rotation = np.array([[np.cos(theta_rad), 0, -np.sin(theta_rad)],
                             [0., 1., 0.],
                             [np.sin(theta_rad), 0, np.cos(theta_rad)]])
        
        self.data_coords = self.data_coords @ rotation.T
        self.cpt_locations =  self.cpt_locations @ rotation[::2,::2].T

    def plot_coordinates(self,original_coordinates = False):
        """plot_coordinates Plots the coordinates of the CPTs

        Args:
            - original_coordinates (bool, optional): If True, plots in original coordinates. Defaults to False.
        """

        if original_coordinates:
            locations = np.array([cpt.coordinates for cpt in self.cpt_list])
        else:
            locations = self.cpt_locations

        plt.plot(locations[:,0],locations[:,1],'dk',label = 'CPT locations')
        plt.xlabel('x-coordinate [m]')
        plt.ylabel('y_coordinate [m]')
        plt.axis('equal')
        plt.show()

class GeostatisticalModel():
    """GeostatisticalModel 

    Attr:
        - nb_dimensions (int):
        - v_dim (int): 
        - gpr (:class:`sklearn.gaussian_process.GaussianProcessRegressor`). Gaussian process regressor. 
        - data_coords (npt.NDArray[np.float64]):
        - data_values (npt.NDArray[np.float64]):
        - noise_level (float): Defaults to None
    
    """        

    def __init__(self,nb_dimensions: int,v_dim: int, kernel = None):
        """__init__ Initiates instance and sets kernel to WhiteKernel() + RBF() if kernel is not provided

        Args:
            - nb_dimensions (int): number of physical dimensions.
            - v_dim (int): Dimension index of the vertical scale of fluctuation
            - kernel (:class:`sklearn.gaussian_process.kernels.Kernel`, optional): Spatial correlation kernel. Defaults to None.

        Raises:
            TypeError: if `kernel` is not a class from module `sklearn.gaussian_process.kernels`

        """            
        self.nb_dimensions = nb_dimensions
        self.v_dim = v_dim

        self.data_coords = np.array([],dtype = np.float64)
        self.data_values = np.array([],dtype = np.float64)
        self.noise_level = None

        # check if `kernel` has the correct parent class
        if kernel: 
            if kernel.__module__ !=  'sklearn.gaussian_process.kernels':
                raise TypeError("`kernel` should an instance of a class from module `sklearn.gaussian_process.kernels`")

        # if kernel does not exist, create a default
        if not kernel:
            length_scale = [100.]*nb_dimensions
            length_scale_bounds = [[1.0,1000.],[1.0,1000.],[1.0,1000.]]
            length_scale[v_dim] = 1.
            length_scale_bounds[v_dim] = [0.01,10.]
            

            kernel = WhiteKernel(0.01) + RBF(length_scale=length_scale,
                                             length_scale_bounds = length_scale_bounds)  

        self.gpr = GPR(kernel = kernel)


    def calibrate(self,coords: npt.NDArray[np.float64],values: npt.NDArray[np.float64]):
        """calibrates the geosatistical model

        Args:
            coords (npt.NDArray[np.float64]): data spatial coordinates
            values (npt.NDArray[np.float64]): data values

        Raises:
            Exception: 
        """ 
        if coords.shape[0] == self.nb_dimensions and coords.shape[1] != self.nb_dimensions:
            self.data_coords = coords.T
            raise Warning('`coords` is transposed to match dimensions (n_points,n_dims)') 
        elif coords.shape[1] == self.nb_dimensions:
            self.data_coords = coords
        else:
            raise Exception("Shape of argument 'X' incompattible with dimensionality:",X.shape) 

        self.data_values = values

        # calibrate
        self.gpr.fit(self.data_coords,self.data_values)


        # extract noise level
        try:
            self.noise_level = self.gpr.kernel_.get_params()['k1__noise_level']
        except:
            self.noise_level = None

        # extracts length scales and transforms to scales of fluctuation
        try:
            gpr_length_scales = self.gpr.kernel_.get_params()['k2__length_scale']
            if 'RBF' in str(self.gpr.kernel):
                self.vertical_scale_fluctuation = gpr_length_scales[self.v_dim] * np.pi / 2.
            elif 'Matern' in str(self.gpr.kernel):
                self.vertical_scale_fluctuation = gpr_length_scales[self.v_dim] * np.sqrt(1/2)
            mask = np.arange(len(gpr_length_scales)) != self.v_dim
            self.anisotropy = list(gpr_length_scales[mask] / gpr_length_scales[self.v_dim])
        except:
            gpr_length_scales = None
            self.vertical_scale_fluctuation = None
            self.anisotropy = None

class ElasticityFieldsFromCpt():
    """ElasticityFieldsFromCpt 
    
    
       Class for the automatic generation of fields of elastic parameters (vs and rho)
    """    
    
    
    def __init__(self,
                 cpt_file_folder:str = './',
                 x_ref:float = 0.,
                 y_ref:float = 0.,
                 orientation_x_axis:float = 0.,
                 based_on_midpoint:bool = False,
                 max_conditioning_points:int = 2000):
                                 
        self.cpt_file_folder = cpt_file_folder
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.orientation_x_axis = orientation_x_axis
        self.based_on_midpoint = based_on_midpoint
        self.max_conditioning_points = max_conditioning_points

        self.conditioning_data = CPT_data(cpt_directory = self.cpt_file_folder)
        self.conditioning_data.read_cpt_data()
        self.conditioning_data.interpret_cpt_data()
        self.conditioning_data.data_coordinate_change(orientation_x_axis = self.orientation_x_axis,based_on_midpoint = self.based_on_midpoint)

        self.thinning_sample_index = np.random.choice(self.conditioning_data.data_coords.shape[0],size = self.max_conditioning_points,replace = False)

        self.trans_model_rho = MarginalTransformation(self.conditioning_data.rho)
        self.trans_model_vs = MarginalTransformation(self.conditioning_data.vs,lognormal = True)


    def calibrate_geostat_model(self,
                          v_dim:int = 1,
                          calibration_indices:tuple = (0,1),
                          seed:int = 13):
        """calibrates the geostatistical model
        Args:
            v_dim (int, optional): vertical dimension. Defaults to 1.
            calibration_indices (tuple, optional): _description_. Defaults to (0,1).
            seed (int, optional): Seed for the `:class:RandomFields`. Defaults to 13.
        """

        ndim_calibrate = len(calibration_indices)

        length_scale_prior = [100.]*ndim_calibrate
        length_scale_prior[v_dim] = 1.

        #kernel = WhiteKernel(0.01) + RBF(length_scale=length_scale_prior)

        self.coord_calibration = np.zeros([self.max_conditioning_points,3])
        for i in calibration_indices:
            self.coord_calibration[:,i] = self.conditioning_data.data_coords[self.thinning_sample_index,i]


        self.geostat_model = GeostatisticalModel(nb_dimensions = ndim_calibrate, v_dim = v_dim)
        self.geostat_model.calibrate(self.coord_calibration[:,calibration_indices],self.trans_model_vs.XtoZ(self.conditioning_data.vs[self.thinning_sample_index]))

        self.rf_vs = RandomFields(model_name = ModelName.Gaussian, 
                            n_dim = 3, 
                            mean = 0, 
                            variance = 1,
                            v_scale_fluctuation = self.geostat_model.vertical_scale_fluctuation, 
                            anisotropy = (self.geostat_model.anisotropy*2)[:2], 
                            angle = [0]*2, 
                            seed = seed,
                            max_conditioning_points=self.max_conditioning_points)
        self.rf_vs.set_conditioning_points(points = self.coord_calibration,
                            values = self.trans_model_vs.XtoZ(self.conditioning_data.vs[self.thinning_sample_index]),
                            noise_level = self.geostat_model.noise_level)
        self.rf_rho = RandomFields(model_name = ModelName.Gaussian, 
                            n_dim = 3, 
                            mean = 0, 
                            variance = 1,
                            v_scale_fluctuation = self.geostat_model.vertical_scale_fluctuation, 
                            anisotropy = (self.geostat_model.anisotropy*2)[:2], 
                            angle = [0]*2, 
                            seed = seed+1,
                            max_conditioning_points=self.max_conditioning_points)
        self.rf_rho.set_conditioning_points(points = self.coord_calibration,
                            values = self.trans_model_rho.XtoZ(self.conditioning_data.rho[self.thinning_sample_index]),
                            noise_level = self.geostat_model.noise_level)


    def generate(self,coordinates):
        """generate random fields

        Args:
            coordinates (numpy.typing.NDArray[np.float64]): 3D coordinated of points to generate random field values for
        """        
        self.rf_vs.generate_conditioned(nodes = coordinates)
        self.rf_rho.generate_conditioned(nodes = coordinates)

        self.vs = self.trans_model_vs.ZtoX(self.rf_vs.conditioned_random_field)
        self.rho = self.trans_model_rho.ZtoX(self.rf_rho.conditioned_random_field)
        self.g0 = self.rho * self.vs**2



if __name__ == '__main__':

    folder = r'.'

    x = np.linspace( -50, 50, 41)
    y = np.linspace(-10, -0, 41)
    z = np.linspace(-20, 20, 41)
    x, y, z = np.meshgrid(x, y, z)
    xy_mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T

    efc = ElasticityFieldsFromCpt(cpt_file_folder = folder,
                                  based_on_midpoint = True,
                                  max_conditioning_points = 1000,
                                  orientation_x_axis = 162)
    efc.calibrate_geostat_model(calibration_indices=(0,1,2))
    
    print(efc.geostat_model.gpr.kernel_)

    %matplotlib qt

    efc.generate(xy_mesh)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = xy_mesh[:,0]
    y = xy_mesh[:,2]
    z = xy_mesh[:,1]
    xd = efc.conditioning_data.data_coords[:,0]
    yd = efc.conditioning_data.data_coords[:,2]
    zd = efc.conditioning_data.data_coords[:,1]
    #%%

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z,c=efc.rho, edgecolors=None, marker="s")
    ax.scatter(xd, yd, zd,c=efc.conditioning_data.rho)
    ax.set_xlabel('x')    
    ax.set_ylabel('y')    
    ax.set_zlabel('z')
    ax.set_zlim([-25,5])
    plt.axis('equal')