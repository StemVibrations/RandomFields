import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl


def plot3D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):
    """
    Plots and saves 3D random field 

    Args:
        - coordinates (list): List of coordinates of the random field
        - random_field (list): List of random field values
        - title (str): Title of the plot
        - output_folder (str): Output folder
        - output_name (str): Output fine name
    """
    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    # rotate the axes so that y is vertical
    ax.view_init(125, -90, 0)

    vmin = np.min([np.min(aux) for aux in random_field])
    vmax = np.max([np.max(aux) for aux in random_field])

    for i, coord in enumerate(coordinates):
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        ax.scatter(x, y, z, c=random_field[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None, marker="s")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_zlabel('z coordinate')

    cax = ax.inset_axes([1.1, 0., 0.05, 1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))
    plt.close()


def plot2D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png",
           colorbar_label: str = '',conditioning_coordinates: list = [], 
           conditioning_values:list = [],figsize: tuple = (6,5)):
    """
    Plots and saves 2D random field

    Args:
        - coordinates (list): List of coordinates of the random field
        - random_field (list): List of random field values
        - title (str): Title of the plot
        - output_folder (str): Output folder
        - output_name (str): Output fine name
        - colorbar_label (str): Label for the colorbar (optional, deafult = '')
        - conditioning_coordinates (list): List of coordinates if the conditioning points (optional, default = [])
        - conditioning_values (list): List of the values at the conditioning points (optional, default = [])
    """

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0.1, 0.1, 0.7, 0.8])

    vmin = np.min([np.min(aux) for aux in random_field])
    vmax = np.max([np.max(aux) for aux in random_field])

    for i, coord in enumerate(coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, c=random_field[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None, marker="s")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')

    for i,coord in enumerate(conditioning_coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, c=conditioning_values[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors='red', linewidth = 0.25 )

    cax = ax.inset_axes([1.1, 0., 0.05, 1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax,label = colorbar_label)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))

    plt.close()


def plot1D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):
    """
    Plots and saves 1D random field

    Args:
    ----------
    coordinates : list
        List of coordinates of the random field
    random_field : list
        List of random field values
    title : str
        Title of the plot
    output_folder : str
        Output folder
    output_name : str
        Output fine name
    """

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_position([0.1, 0.1, 0.7, 0.8])

    vmin = np.min([np.min(aux) for aux in random_field])
    vmax = np.max([np.max(aux) for aux in random_field])

    color = 'black'
    if len(coordinates) > 1:
        color = 'gray' 

    for i, coord in enumerate(coordinates):
        x = coord[:, 0]
        y = np.array(random_field[i]).ravel()
        ax.plot(x, y, color = color)
    
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('random field value')

    plt.tight_layout()

    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))
    plt.close()    