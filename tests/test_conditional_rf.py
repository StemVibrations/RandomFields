import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest

from random_fields.geostatistical_cpt_interpretation import CPT_data, MarginalTransformation, GeostatisticalModel
from random_fields.generate_field import RandomFields, ModelName

SHOW_RESULTS = False



@pytest.mark.parametrize("cpt_folder_path, test_file", [
    ("./tests/cpts/gef", "./tests/data/2d_rf_gef.pickle"),
    ("./tests/cpts/xml", "./tests/data/2d_rf_xml.pickle")
])
def test_2d_random_field(cpt_folder_path, test_file):
    """
    Test the generation of a conditional 2D random field with GEF & XML files.

    Args:
        - cpt_folder_path (str): path to the folder containing the CPT files.
        - test_file (str): path to the file containing the expected results.
    """

    cpt_folder = Path(cpt_folder_path)
    cpt_data = CPT_data(cpt_directory=cpt_folder)
    cpt_data.read_cpt_data()
    cpt_data.interpret_cpt_data()

    cpt_data.plot_coordinates(show=False)
    assert os.path.exists("coordinates.png")
    os.remove("coordinates.png")

    # transform the data to a modelling domain
    cpt_data.data_coordinate_change(orientation_x_axis=73, based_on_midpoint=True)

    marginal_transformator = MarginalTransformation(cpt_data.vs, min_value=50)
    marginal_transformator.plot(x_label='$u$ : standard-normal variable', y_label='$v$ : shear wave velocity [m/s]')

    np.random.seed(14)
    index_selection = np.random.choice(len(cpt_data.vs), size=2000, replace=False)
    coords = cpt_data.data_coords[index_selection]
    z_data = marginal_transformator.x_to_z(x=cpt_data.vs[index_selection])

    geo_model = GeostatisticalModel(nb_dimensions=2, v_dim=1)
    geo_model.calibrate(coords=coords[:, [2, 1]], values=z_data)

    random_field_generator = RandomFields(model_name=ModelName.Gaussian,
                                        n_dim=2,
                                        mean=0,
                                        variance=1,
                                        v_scale_fluctuation=geo_model.vertical_scale_fluctuation,
                                        anisotropy=geo_model.anisotropy,
                                        angle=[0],
                                        seed=14)

    I = np.random.choice(len(cpt_data.vs), size=500, replace=False)

    coords = cpt_data.data_coords[I]
    values = cpt_data.vs[I]
    random_field_generator.set_conditioning_points(points=coords[:, [2, 1]],
                                                values=marginal_transformator.x_to_z(x=values),
                                                noise_level=geo_model.noise_level)

    # create grit of points on the domnain (-220,220) by (-24,-1) to generate a field for.
    x = np.linspace(-220, 220, 250)
    z = np.linspace(-24, -1, 250)
    X, Z = np.meshgrid(x, z)

    # generate a conditioned random field
    sample_coords = np.array([X.ravel(), Z.ravel()]).T
    random_field_generator.generate_conditioned(nodes=sample_coords)

    # Transform the generated standard-normal field to the distribution of the shear wave velocity
    z_map = random_field_generator.conditioned_random_field
    vs_map = marginal_transformator.z_to_x(z_map[:250 * 250].reshape([250, 250]))

    # check the results
    with open(test_file, "rb") as f:
        vs_test = pickle.load(f)

    assert np.allclose(vs_map, vs_test)

    if SHOW_RESULTS:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.contourf(X, Z, vs_map, vmin=50, vmax=400)
        scatter = plt.scatter(cpt_data.data_coords[:, 2], cpt_data.data_coords[:, 1], c=cpt_data.vs, vmin=50, vmax=400)
        plt.colorbar(scatter, ax=ax, label="Vs [m/s]")
        plt.show()
