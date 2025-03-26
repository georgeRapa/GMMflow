import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import art3d
import matplotlib.transforms as transforms

def confidence_ellipsoid(ax, cov, mean=np.array([0, 0, 0]), n_std=3.0, facecolor='None', edgecolor='black', **kwargs):

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]
    # calculating the stdandard deviation of z ...
    scale_z = np.sqrt(cov[2, 2]) * n_std
    mean_z = mean[2]

    # Make data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = scale_x* np.outer(np.cos(u), np.sin(v)) + mean_x
    y = scale_y* np.outer(np.sin(u), np.sin(v)) + mean_y
    z = scale_z* np.outer(np.ones(np.size(u)), np.cos(v)) + mean_z
    # Plot the surface
    ax.plot_surface(x, y, z, edgecolor='black', alpha=0.05, rstride=1, cstride=1,
                        linewidth=0.01, antialiased=True)


def confidence_ellipse(ax, cov, mean=np.zeros(2), n_std=3.0, facecolor='None', edgecolor='black', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    cov : array_like, shape (n,n)
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    # if x.size != y.size:
    #     raise ValueError("x and y must be the same size")

    # cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor, edgecolor=edgecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)