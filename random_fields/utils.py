import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl


def plot3D(coordinates: list, random_field: list, title: str = "Random Field",
           output_folder = "./", output_name: str = "random_field.png"):
    """
    Plot 3D random field

    Parameters
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
    fig = plt.figure(1, figsize=(6, 5))
    plt.rcParams['font.sans-serif'] = 'Arial'  # Set a common font
    plt.rcParams['pdf.fonttype'] = 42  # Ensure that fonts are embedded in PDF/EPS
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
           output_folder = "./", output_name: str = "random_field.png"):
    """
    Plot 2D random field

    Parameters
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
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams['font.sans-serif'] = 'Arial'  # Set a common font
    plt.rcParams['pdf.fonttype'] = 42  # Ensure that fonts are embedded in PDF/EPS
    ax.set_position([0.1, 0.1, 0.7, 0.8])

    vmin = np.min([np.min(aux) for aux in random_field])
    vmax = np.max([np.max(aux) for aux in random_field])

    for i, coord in enumerate(coordinates):
        x, y = coord[:, 0], coord[:, 1]
        ax.scatter(x, y, c=random_field[i], vmin=vmin, vmax=vmax, cmap="viridis", edgecolors=None, marker="s")

    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')

    cax = ax.inset_axes([1.1, 0., 0.05, 1])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, cax=cax)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_folder, output_name))
    plt.close()