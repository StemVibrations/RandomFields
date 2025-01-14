import os
import pickle
import numpy as np
import pytest
from scipy.stats import norm
from random_fields.generate_field import RandomFields, ModelName
from random_fields.geostatistical_cpt_interpretation import ElasticityFieldsFromCpt, RandomFieldProperties
from random_fields.utils import plot3D


def test_model_names():
    "test model names"
    rf = RandomFields(ModelName.Gaussian, 2, 2, 2, 2, [2], [2])
    assert rf.random_field_model.__name__ == 'Gaussian'

    rf = RandomFields(ModelName.Exponential, 2, 2, 2, 2, [2], [2])
    assert rf.random_field_model.__name__ == 'Exponential'

    rf = RandomFields(ModelName.Matern, 2, 2, 2, 2, [2], [2])
    assert rf.random_field_model.__name__ == 'Matern'

    rf = RandomFields(ModelName.Linear, 2, 2, 2, 2, [2], [2])
    assert rf.random_field_model.__name__ == 'Linear'


def test_failure_ndim():
    "test failure with number of dimensions"
    with pytest.raises(ValueError):
        RandomFields(ModelName.Gaussian, 4, 2, 2, 2, [2], [2])


def test_model_name_failure():
    """test failure with wrong model name"""
    with pytest.raises(AttributeError):
        RandomFields(ModelName.ASD)


def test_distribution_RF_struc():
    """test distribution of 2D random field with structured mesh"""
    nb_runs = 100

    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 50, 50)

    x, y = np.meshgrid(x, y)

    rf_emsamble = []
    param = []
    for i in range(nb_runs):
        rf = RandomFields(ModelName.Gaussian, 2, 10, 2, 1, [1], [1], seed=i)
        rf.generate(np.array([x.ravel(), y.ravel()]).T)
        rf_emsamble.append(rf.random_field.field)
        mu_un, std_un = norm.fit(rf.random_field.field)
        param.append([mu_un, std_un])

    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 0]), 10, decimal=2)
    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 1])**2, 2, decimal=2)


def test_distribution_RF_struc_3D():
    """test distribution of 2D random field with structured mesh"""
    nb_runs = 100

    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 40, 10)
    z = np.linspace(0, 15, 15)

    x, y, z = np.meshgrid(x, y, z)

    rf_emsamble = []
    param = []
    for i in range(nb_runs):
        rf = RandomFields(ModelName.Gaussian, 3, 10, 2, 1, [1, 0.5], [1, 1], seed=i)
        rf.generate(np.array([x.ravel(), y.ravel(), z.ravel()]).T)
        rf_emsamble.append(rf.random_field.field)
        mu_un, std_un = norm.fit(rf.random_field.field)
        param.append([mu_un, std_un])

    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 0]), 10, decimal=2)
    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 1])**2, 2, decimal=2)


def test_distribution_RF_unstruc():
    """test distribution of 2D random field with unstructured mesh"""

    with open("./tests/data/mesh_2D.pickle", "rb") as fo:
        nodes_fields = pickle.load(fo)

    nb_runs = 100

    x = np.array(nodes_fields)[:, 0]
    y = np.array(nodes_fields)[:, 1]

    rf_ensamble = []
    param = []
    for i in range(nb_runs):
        rf = RandomFields(ModelName.Gaussian, 2, 10, 2, 0.05, [1], [1], seed=i)
        rf.generate(np.array([x.ravel(), y.ravel()]).T)
        rf_ensamble.append(rf.random_field.field)
        mu_un, std_un = norm.fit(rf.random_field.field)
        param.append([mu_un, std_un])

    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 0]), 10, decimal=2)
    np.testing.assert_array_almost_equal(np.mean(np.array(param)[:, 1])**2, 2, decimal=2)


def test_conditioned_RF_3D_mean_variance():
    """test the conditioned random field for correct mean and variance"""

    # mesh coordinates
    x = np.linspace(0, 100, 5)
    y = np.linspace(0, 50, 5)
    z = np.linspace(0, 25, 5)
    x, y, z = [i.ravel() for i in np.meshgrid(x, y, z)]

    # random field properties
    nb_dimensions = 3
    mean = 10
    variance = 2
    vertical_scale_fluctuation = 10
    anisotropy = [2.5, 2.5]
    angle = [0, 0]
    model_rf = ModelName.Gaussian

    # generate and plot random field
    rf = RandomFields(model_rf, nb_dimensions, mean, variance, vertical_scale_fluctuation, anisotropy, angle, seed=14)
    rf.generate(np.array([x, y, z]).T)

    # declare conditioning points
    xc = np.array([50.] * 4)
    yc = np.linspace(0, 50, 4)
    zc = np.array([25] * 4)
    vc = np.array([15] * 4)

    rf.set_conditioning_points(np.array([xc, yc, zc]).T, vc, noise_level=0.01)

    # generate conditioned random field model
    rf.generate_conditioned(np.array([x, y, z]).T)

    # load reference solution
    data_ref = np.loadtxt('./tests/data/conditioned_rf_3D.txt')
    kriging_mean_ref = data_ref[0]
    kriging_std_ref = data_ref[1]
    kriging_field_ref = data_ref[2]

    # test difference in mean, std and single realisation
    np.testing.assert_array_almost_equal(kriging_mean_ref, rf.kriging_mean, decimal=4)
    np.testing.assert_array_almost_equal(kriging_std_ref, rf.kriging_std, decimal=4)
    np.testing.assert_array_almost_equal(kriging_field_ref, rf.conditioned_random_field, decimal=4)


def test_conditioned_RF_helper_3D():
    """
    Test the ElasticityFieldsFromCpt class
    """

    orientation_x_axis = 72.
    cpt_folder = "./tests/cpts/gef"
    elastic_field_generator_cpt = ElasticityFieldsFromCpt(cpt_file_folder=cpt_folder,
                based_on_midpoint = True,
                max_conditioning_points = 1000,
                orientation_x_axis = orientation_x_axis,
                porosity = 0.3,
                water_density = 1000,
                return_property = [RandomFieldProperties.YOUNG_MODULUS, RandomFieldProperties.DENSITY_SOLID],
                )
    elastic_field_generator_cpt.calibrate_geostat_model()

    # create grit of points on the domnain (-220,220) by (-24,-1) to generate a field for.
    import numpy as np
    x = np.linspace(-5, 5, 21)
    y = np.linspace(-25, 0, 26)
    z = np.linspace(0, 30, 31)
    X, Y, Z = np.meshgrid(x, y, z)

    elastic_field_generator_cpt.generate(np.array([X.ravel(), Y.ravel(), Z.ravel()]).T)

    plot3D([np.array([X.ravel(), Y.ravel(), Z.ravel()]).T], [elastic_field_generator_cpt.generated_field[0]],
            title="Random Field",
           output_folder="./",
           output_name="random_field_3D.png",
           figsize=(10, 10),
           conditional_points=[elastic_field_generator_cpt.coordinates_sampled_conditioning,
                               elastic_field_generator_cpt.conditioning_sampled_data[0]],
           )

    with open("./tests/data/conditional_random_field_3D.pickle", "rb") as fi:
        data_org = pickle.load(fi)

    assert os.path.isfile("random_field_3D.png")
    os.remove("random_field_3D.png")
    np.testing.assert_allclose(data_org[0], elastic_field_generator_cpt.generated_field[0], rtol=1e-3)
    np.testing.assert_allclose(data_org[1], elastic_field_generator_cpt.generated_field[1], rtol=1e-3)

def test_RF_properties():
    """
    Test the RandomFieldProperties enum
    """
    # Check that the enum members are instances of RandomFieldProperties
    assert isinstance(RandomFieldProperties.YOUNG_MODULUS, RandomFieldProperties)
    assert isinstance(RandomFieldProperties.DENSITY_SOLID, RandomFieldProperties)

    # Check that the names match correctly
    assert RandomFieldProperties.YOUNG_MODULUS.name == "YOUNG_MODULUS"
    assert RandomFieldProperties.DENSITY_SOLID.name == "DENSITY_SOLID"
