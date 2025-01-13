import os
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import gstools as gs


def plot3D(coordinates: List[npt.NDArray[np.float64]],
           random_field: List[gs.field.srf.SRF],
           title: str = "Random Field",
           output_folder="./",
           output_name: str = "random_field.png",
           show: bool = False):
    """
    Plots and saves 3D random field

    Args:
        - coordinates (List[npt.NDArray[np.float64]]): List of coordinates of the random field
        - random_field (List[gs.field.srf.SRF]): List of random field values
        - title (str): Title of the plot
        - output_folder (str): Output folder
        - output_name (str): Output fine name
        - show (bool): Show the plot (optional, default = False)
    """
    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    # rotate the axes so that y is vertical
    ax.view_init(125, -90, 0)

    rfield = np.array(random_field)
    vmax = np.max(rfield)
    vmin = np.min(rfield)

    for i, coord in enumerate(coordinates):
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        ax.scatter(x, y, z, c=rfield[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None, marker="s")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_zlabel('z coordinate')

    cax = ax.inset_axes((1.1, 0., 0.05, 1))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))

    if show:
        plt.show()
    plt.close()


def plot2D(coordinates: List[npt.NDArray[np.float64]],
           random_field: List['gs.field.srf.SRF'],
           title: str = "Random Field",
           output_folder="./",
           output_name: str = "random_field.png",
           colorbar_label: str = '',
           conditioning_coordinates: List[npt.NDArray[np.float64]] = [],
           conditioning_values: List[npt.NDArray[np.float64]] = [],
           figsize: Tuple[float, float] = (6, 5),
           show: bool = False):
    """
    Plots and saves 2D random field

    Args:
        - coordinates (List[npt.NDArray[np.float64]]): List of coordinates of the random field
        - random_field (List[gs.field.srf.SRF]): List of random field values
        - title (str): Title of the plot
        - output_folder (str): Output folder
        - output_name (str): Output fine name
        - colorbar_label (str): Label for the colorbar (optional, deafult = '')
        - conditioning_coordinates (List[npt.NDArray[np.float64]]): List of coordinates if the conditioning points
        (optional, default = [])
        - conditioning_values (List[npt.NDArray[np.float64]]): List of the values at the conditioning points
          (optional, default = [])
        - figsize (Tuple[float, float]): Figure size (optional, default = (6, 5))
        - show (bool): Show the plot (optional, default = False)
    """

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position((0.1, 0.1, 0.7, 0.8))

    rfield = np.array(random_field)
    vmax = np.max(rfield)
    vmin = np.min(rfield)

    for i, coord in enumerate(coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, c=rfield[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None, marker="s")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')

    for i, coord in enumerate(conditioning_coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x,
                   y,
                   c=conditioning_values[i],
                   vmin=vmin,
                   vmax=vmax,
                   cmap="viridis",
                   edgecolors='red',
                   linewidth=0.25)

    cax = ax.inset_axes((1.1, 0., 0.05, 1))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax, label=colorbar_label)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))

    if show:
        plt.show()
    plt.close()


def plot1D(coordinates: List[npt.NDArray[np.float64]],
           random_field: List[gs.field.srf.SRF],
           title: str = "Random Field",
           output_folder="./",
           output_name: str = "random_field.png",
           conditional_points: Optional[List[npt.NDArray[np.float64]]] = None,
           show: bool = False):
    """
    Plots and saves 1D random field

    Args:
        - coordinates (List[npt.NDArray[np.float64]]): List of coordinates of the random field
        - random_field (List[gs.field.srf.SRF]): List of random field values
        - title (str): Title of the plot
        - output_folder (str): Output folder
        - output_name (str): Output fine name
        - conditional_points (List[npt.NDArray[np.float64]]): List of the values at the conditioning points, and
            kriging mean and standard deviation (optional, default = None)
        - show (bool): Show the plot (optional, default = False)
    """

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_position((0.1, 0.1, 0.7, 0.8))

    color = 'black'
    if len(coordinates) > 1:
        color = 'gray'

    for i, coord in enumerate(coordinates):
        x = coord[:, 0]
        y = np.array(random_field[i]).ravel()
        ax.plot(x, y, color=color)

    if conditional_points:
        plt.plot(x, conditional_points[2], label='kriged field', color='k')
        plt.fill_between(x.ravel(),
                         conditional_points[2] - 1.65 * conditional_points[3],
                         conditional_points[2] + 1.65 * conditional_points[3],
                         label='90% uncertainty bound',
                         color='gray',
                         alpha=0.5)
        plt.scatter(conditional_points[0], conditional_points[1], label='conditioning points', color='k')
        plt.legend()

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('random field value')

    plt.tight_layout()

    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))

    if show:
        plt.show()
    plt.close()
