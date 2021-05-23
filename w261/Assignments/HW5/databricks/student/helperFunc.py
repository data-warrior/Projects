# Databricks notebook source
#!/opt/anaconda/bin/python
"""
This file contains helper functions for generating, transforming
and plotting 2 dimensional data to use in testing & for ML demos.

Avaliable functions include:
    augment(X)
    plot2DModels(data, models=[], names = [], title=None) returns figx
    plotErrorSurface(data, weight_grid, loss, title=None) returns figx

"""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def augment(X):
    """
    Takes an np.array whose rows are data points and augments each
    each row with a 1 in the last position to represent the bias.
    """
    return np.insert(X, -1, 1.0, axis=1)


def plot2DModels(data, models=[], names = [], title=None):
    """
    Plot a set of 2d models for comparison.
    INPUT:  data       - numpy array of points x, y
            model_list - [(label,[w_0, w_1]), ...]
            title      - (optional) figure title
    """
    # create plot
    fig, ax = plt.subplots()
    # plot data
    ax.plot(data[:,0], data[:,1],'o')
    domain = [min(data[:,0]), max(data[:,0])]
    # plot models
    for W,label in zip(models, names):
        m , b = W[0], W[1]
        yvals = [m*x + b for x in domain]
        ax.plot(domain, yvals, linewidth=1, label=label)
    if models:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # title & formatting
    if title:
        plt.title(title)

    figx = plt.show()
    return figx

def plotErrorSurface(data, weight_grid, loss, title=None):
    """
    Plot a set of 2d models for comparison.
    INPUT:  data    - numpy array of points x, y
            weight_grid  - numpy array of weight vectors [w_0, w_1]
            loss    - list/array of loss corresponding to ^
            title   - (optional) figure title
    """
    # create figure
    fig = plt.figure(figsize=plt.figaspect(0.25))

    # plot data
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(data[:,0], data[:,1],'o')
    ax1.set_title("Linear Models in 2D", fontsize=14)
    plt.xlabel('Input Data')
    plt.ylabel('Output Value')
    domain = [min(data[:,0]), max(data[:,0])]
    
    # plot models
    for idx, W in enumerate(weight_grid):
        m , b = W[0], W[1]
        yvals = [m*x + b for x in domain]
        ax1.plot(domain, yvals, linewidth=1)
        
    # plot loss in 3D
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title("Loss as a function of weights.", fontsize=15)
    plt.xlabel('W_0')
    plt.ylabel('W_1')
    X,Y = weight_grid.T
    ax2.scatter(X,Y,loss, c=loss, cmap=cm.rainbow)
    
    
    # plot error surface in 3D
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.set_title("Inferred Error Surface", fontsize=15)
    plt.xlabel('W_0')
    plt.ylabel('W_1')
    X,Y = weight_grid.T
    surf = ax3.plot_trisurf(X,Y,loss, cmap=cm.rainbow, 
                            linewidths = 2.0, alpha=0.65)
    fig.colorbar(surf, alpha=0.65, shrink = 0.5)
    
    # title & formatting
    if title:
        plt.title(title)
    
    figx = plt.show()
    return figx      
