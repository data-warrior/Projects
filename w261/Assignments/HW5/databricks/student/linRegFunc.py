# Databricks notebook source
#!/opt/anaconda/bin/python
"""
This file contains several helper functions for OLS Linear Regression.

Avaliable functions include:
    OLSloss(X, y, model)
    OLSGradient(X, y, model)
    GDUpdate(X, y, nIter, init_model, learning_rate, verbose = False)
    plotGDProgress(data, models, loss, loss_fxn = OLSLoss, show_contours = True)
    SGDUpdate(X, y, B, nIter, init_model, learning_rate, verbose = False)
    mean_absolute_percentage_error(y_true, y_pred)
"""
# general imports
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# custom module import
from helperFunc import augment


def OLSLoss(X, y, model):
    """
    Computes mean squared error for a linear model.
    INPUT:  X - numpy array (each row = augmented input point)
            y - numpy array of true outputs
            model  - [w_0, w_1] (coefficient & bias)
    """
    N = len(X)
    W = np.array(model)
    return 1/float(N) * sum((W.dot(X.T) - y)**2)
    
def OLSGradient(X, y, model):
    """
    Computes the gradient of the OLS loss function for 
    the provided data & linear model.
    INPUT:  X - numpy array (each row = augmented input point)
            y - numpy array of true outputs
            model  - [w_0, w_1] (coefficient & bias)
    """
    N = len(X)
    W = np.array(model)
    return 2.0/N *(W.dot(X.T) - y).dot(X)

def GDUpdate(X, y, nIter, init_model, learning_rate, verbose = False):
    """
    Performs Gradient Descent Updates for linear models using OLS Loss.
    INPUT:  X - numpy array (each row = augmented input point)
            y - numpy array of true outputs
            nIter - number of updates to perform
            init_model  - [w_0, w_1] starting coefficient & bias
            learning_rate - step size for the update
            verbose - (optional) printout each update
    OUTPUT: models , loss  - two lists
    """
    # keep track of our progress
    models = [init_model]
    loss = [OLSLoss(X,y,init_model)]
    
    # perform updates
    for idx in range(nIter):
        gradient = OLSGradient(X, y, models[-1])
        
        update = np.multiply(gradient,learning_rate)
        
        new_model = models[-1] - update
        
        if verbose:
            print(f'Model {idx}: [{models[-1][0]:.2f}, {models[-1][1]:.2f}]')
            print(f'Loss: {loss[-1]}')
            print(f'     >>> gradient: {gradient}')
            print(f'     >>> update: {update}')
        models.append(new_model)
        loss.append(OLSLoss(X,y,new_model))
    if verbose:
            print(f'Model {nIter}: [{models[-1][0]:.2f}, {models[-1][1]:.2f}]')
            print(f'Loss: {loss[-1]}')
    return np.array(models), loss


def SGDUpdate(X, y, nIter, B, init_model, learning_rate, verbose = False):
    """
    WARNING: SGD should be choosing points at random. mini-batch should be shuffling the data at each iteration!
             This is an oversimlified implementation without any randomness/shuffling
    
    
    Performs Stoichastic Gradient Descent Updates for linear models using OLS Loss.
    INPUT:  X - numpy array (each row = augmented input point)
            y - numpy array of true outputs
            nIter - number of updates to perform
            B - batchsize (integer)
            init_model  - [w_0, w_1] starting coefficient & bias
            learning_rate - step size for the update
            verbose - (optional) printout each update
    OUTPUT: models , loss  - two lists
    """
    # keep track of our progress
    models = [init_model]
    loss = [OLSLoss(X,y,init_model)]
    
    # perform updates
    n = len(X)
    for idx in range(nIter):
        j = (idx*B)%n # index to start batch
        batch_X, batch_y = X[j:j+B], y[j:j+B]
                
        gradient = OLSGradient(batch_X, batch_y, models[-1])
        update = np.multiply(gradient,learning_rate)
        new_model = models[-1] - update
        
        if verbose:
            print(f'Model {idx}: [{models[-1][0]:.2f}, {models[-1][1]:.2f}]')
            print(f'Loss: {loss[-1]}')
            print(f'     >>> gradient: {gradient}')
            print(f'     >>> update: {update}')
        models.append(new_model)
        loss.append(OLSLoss(X,y,new_model))
    
    if verbose:
            print(f'Model {nIter}: [{models[-1][0]:.2f}, {models[-1][1]:.2f}]')
            print(f'Loss: {loss[-1]}')
    return np.array(models), loss


def plotGDProgress(data, models, loss, loss_fxn = OLSLoss, show_contours = True):    
    """
    Plot a set of 2d models for comparison.
    INPUT:  data    - numpy array of points x, y
            models  - numpy array of weight vectors [w_0, w_1]
            loss    - list/array of loss corresponding to ^
            title   - (optional) figure title
    """
    # Create figure w/ two subplots
    fig = plt.figure(figsize=plt.figaspect(0.35))
    ax1 = plt.subplot(1, 2, 1)
    ax1.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2 = plt.subplot(1, 2, 2)
    ax2.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    fig.subplots_adjust(wspace=.6)
    
    ##### Problem Domain Space #####
    
    # plot data
    ax1.plot(data[:,0], data[:,1],'o')
    ax1.set_title("Problem Domain Space", fontsize=18)
    ax1.set_xlabel('Input Data')
    ax1.set_ylabel('Output Value')
    domain = [min(data[:,0]), max(data[:,0])]
    
    # plot models
    for idx, W in enumerate(models):
        m , b = W[0], W[1]
        yvals = [m*x + b for x in domain]
        name = 'm%s:%.2f' %(idx, loss[idx])
        ax1.plot(domain, yvals, label=name, linewidth=1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    ##### Model Parameter Space #####
    
    # plot loss for our models
    w0, w1 = models.T
    ax2.plot(w0,w1)
    ax2.plot(w0,w1, 's')
    ax2.set_title("Model Parameter Space", fontsize=18)
    ax2.set_xlabel('W_0 (slope)')
    ax2.set_ylabel('W_1 (intercept)')
    
    # plot contour lines
    if show_contours:
        # grid parameters -- just a bit larger than models
        w0_min, w0_max= min(w0)*0.9, max(w0)*1.1
        w1_min, w1_max = min(w1)*0.9, max(w1)*1.2
        w0_step = (w0_max - w0_min)/20
        w1_step = (w1_max - w1_min)/20 
        # create loss grid for contour plot
        grid_w0, grid_w1 = np.mgrid[w0_min:w0_max:w0_step,
                                    w1_min:w1_max:w1_step]
        grid_loss = [loss_fxn(augment(data)[:,:2], data[:,1], model)
                     for model in zip(grid_w0.flatten(), grid_w1.flatten())]
        grid_loss = np.array(grid_loss).reshape(20,20)
        # plot loss contours
        topo_levels = np.logspace(min(np.log(min(loss)),0.1), 
                                  min(np.log(max(loss))/10,20))
        CS = ax2.contour(grid_w0, grid_w1, grid_loss, 
                         levels = topo_levels, cmap = 'rainbow', 
                         linewidths = 2.0, alpha=0.35)

    figx = plt.show()
    return figx    
    
def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Use of this metric is not recommended because can cause division by zero
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """

    return np.mean(np.abs((y_true.ravel() - y_pred.ravel()) / y_true.ravel())) * 100 
