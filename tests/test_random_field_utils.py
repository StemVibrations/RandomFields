import numpy as np
import pytest
import os
import sys
from random_fields.generate_field import RandomFields, ModelName
from random_fields.utils import plot2D, plot3D


@pytest.fixture(scope="function")
def cleanup_generated_files():
    # This fixture runs before the test
    yield
    # This code will run after the test
    os.remove("random_field.eps")


def test_distribution_RF_struc(cleanup_generated_files):
    """test distribution of 2D random field with structured mesh"""

    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 50, 50)

    x, y = np.meshgrid(x, y)

    rf = RandomFields(ModelName.Gaussian, 2, 10, 2, 1, [1], [1], seed=14)
    rf.generate(np.array([x.ravel(), y.ravel()]).T)

    plot2D([np.array([x.ravel(), y.ravel()]).T], [rf.random_field],
           title="Random Field",
           output_folder="./",
           output_name="random_field.eps")

    with open("./tests/data/random_field.eps", "r") as fi:
        data_org = fi.read().splitlines()

    with open("./random_field.eps", "r") as fi:
        data_new = fi.read().splitlines()

    header = 5
    idx_end = data_org.index("currentfile DataString readhexstring pop")
    data = [val == data_new[header + i] for i, val in enumerate(data_org[header:idx_end])]
    assert all(data)

def test_distribution_RF_struc_3D(cleanup_generated_files):
    """test distribution of 3D random field with structured mesh"""

    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 40, 10)
    z = np.linspace(0, 15, 15)

    x, y, z = np.meshgrid(x, y, z)

    rf = RandomFields(ModelName.Gaussian, 3, 10, 2, 1, [1, 0.5], [1, 1], seed=14)
    rf.generate(np.array([x.ravel(), y.ravel(), z.ravel()]).T)

    plot3D([np.array([x.ravel(), y.ravel(), z.ravel()]).T], [rf.random_field],
           title="Random Field",
           output_folder="./",
           output_name="random_field.eps")

    if sys.platform == "win32":
        file_test = "./tests/data/random_field_3D_windows.eps"
    elif sys.platform == "linux":
        file_test = "./tests/data/random_field_3D_linux.eps"
    else:
        raise Exception("Platform not supported")

    with open(file_test, "r") as fi:
        data_org = fi.read().splitlines()

    with open("./random_field.eps", "r") as fi:
        data_new = fi.read().splitlines()

    header = 5
    idx_end = data_org.index("currentfile DataString readhexstring pop")
    data = [val == data_new[header + i] for i, val in enumerate(data_org[header:idx_end])]
    assert all(data)
