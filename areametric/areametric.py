'''
    : --------------------------------------- #
    Created: Oct 2020
    Edited:  Mar 2022
    
    Marco De Angelis & Jaleena Sunny
    
    web: github.com/marcodeangelis
    org:University of Liverpool
    
    GNU General Public License v3.0
    : --------------------------------------- #

    Computes the area metric between two dataseries or the area metric of a mixture.
    
    This version works with data sets of different sizes.

    When datasets have same size the code is fastest.

'''
from __future__ import annotations
from typing import Union
from matplotlib.pyplot import axis

import numpy as np

from numpy import (ndarray, concatenate, linspace, diff, argmax, argmin, arange, transpose, prod, empty)

from .dataseries import (dataseries, mixture, mixture_given_dimension_index, map_index_flat_to_array)
from .methods import (ecdf, ecdf_p, inverse_quantile_function, inverse_quantile_mixture, quantile_function, is_compatible)


def areame_algorithm(x_: ndarray, y_: ndarray) -> float: # inputs lists of doubles # numpy array are not currently supported.
    '''
    : --------------------------- ∞ 

    cre: Wed Oct 7 2020
    edi: Tue Mar 8 2022

    web: github.com/marcodeangelis 
    org: University of Liverpool 

    MIT

    : --------------------------- ∞

    Computes the area metric between two dataseries of dimension 1. A dataseries of dimension 1 is a 1d-array.

    Works also on samples of different sizes.

    When samples have same size or one size is a multiple of the other the code is fastest.

    Inputs
    ------
    x : ndarray
        An array-like structure of single dimension, e.g. a list or a Numpy array.
    y : ndarray
        An array-like structure of single dimension, e.g. a list or a Numpy array.

    Output
    ------
    output : float
        The area metric value.
    '''
    x, y = dataseries(x_), dataseries(y_) 
    n1, n2 = len(x), len(y)
    n = n1 if n1>n2 else n2
    qx, qy = quantile_function(x), quantile_function(y)
    if ((n1<n2) & (n2%n1!=0)) | ((n2<n1) & (n1%n2!=0)): # slow branch terminates in context
        x_value = x.value
        y_value = y.value
        xysv = concatenate((x_value,y_value))
        xysv.sort()
        uu = diff(xysv) # steps width
        xv_inverse_quantile = x_value[x.index].searchsorted(xysv[:-1], 'right')/n1 # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        yv_inverse_quantile = y_value[y.index].searchsorted(xysv[:-1], 'right')/n2 # https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        return np.sum(np.multiply(np.abs(xv_inverse_quantile - yv_inverse_quantile), uu))
    elif (n2>n1) & (n2%n1==0): p = ecdf_p(y)
    elif (n1==n2) | ((n1>n2) & (n1%n2==0)): p = ecdf_p(x)
    p_= concatenate(([0.],p)) 
    pm = (p+p_[:-1])/2 # mid height of each step
    return sum(abs(qx(pm) - qy(pm)) / n)
        
def areame_tensor(x:ndarray,y:ndarray) -> ndarray: # x and y must have the same dimension (not shape!)
    """
    Inputs
    ------
    x : ndarray
        An array-like structure of any dimension, e.g. a list or a Numpy array.
    y : ndarray
        An array-like structure of any dimension, e.g. a list or a Numpy array.

    Output
    ------
    areame_sorted_reshape : ndarray
        An ndarray containing the values of the area metric arranged in an array of shape equal to the dimension of x and y.
    """
    x_shape, y_shape = x.shape, y.shape
    rep_x, rep_y = x_shape[0], y_shape[0]
    dim_xy = x_shape[1:] # must be = y_shape[:1]
    permut_x, permut_y = list(arange(len(x_shape))), list(arange(len(y_shape))) # order of dimension
    permutx_rep_last, permuty_rep_last = permut_x[1:]+[permut_x[0]], permut_y[1:]+[permut_y[0]] # new order of dimensions with dimension 0 moved to last
    xt, yt = transpose(x,permutx_rep_last), transpose(y,permuty_rep_last) # array transposed with rep dimension as last
    m=prod(dim_xy,dtype=int)
    areame_sorted_flat = empty((m,))
    for j in range(m): # loop over all elements except the first dimension
        start_x,end_x = int(j*rep_x), int((j+1)*rep_x)
        start_y,end_y = int(j*rep_y), int((j+1)*rep_y)
        x_flat = xt.flatten()[start_x:end_x]
        y_flat = yt.flatten()[start_y:end_y]
        areame_sorted_flat[j] = areame_algorithm(x_flat,y_flat) # here deploy area-metric algorithm
    areame_sorted_reshape = areame_sorted_flat.reshape(dim_xy) # from flat back to shape1
    return areame_sorted_reshape # if values are already sorted (increasing) this must return x

def areaMe(x_:ndarray,y_:ndarray) -> Union[ndarray,float]:
    x, y = dataseries(x_), dataseries(y_) 
    if is_compatible(x,y)==False: raise TypeError('The area metric can only be computed between compatible data structures.\nCompatible ds are ds with the same dimension.')
    if x.tabular: return areame_tensor(x,y) # or y.tabular==True
    else: return areame_algorithm(x,y)

def area_chunks(x_:ndarray,y_:ndarray) -> ndarray: # not yet tensor
    x, y = dataseries(x_), dataseries(y_) 
    iqx, iqy = inverse_quantile_function(x,side='right'), inverse_quantile_function(y,side='right')
    xy = x + y # concatenate the two dataseries # sort happens here again
    xysv = xy.value_sorted # get data after sorting
    u = abs(iqx(xysv) - iqy(xysv)) # steps height
    v = diff(xysv) # steps width
    return u[:-1]*v # area chunks

def areame_mixture_inneralgorithm(x,one_sorted_values): 
    '''
    Computes the area metric between 1d mixtures.
    '''
    iqx_r = inverse_quantile_mixture(x,one_sorted_values,side='right') # inverse quantile value for each sample in the mixture.
    uu = diff(one_sorted_values,) # steps width
    vv = abs(np.max(iqx_r,axis=0) - np.min(iqx_r,axis=0)) # steps height
    return np.sum((uu*vv[:-1]),axis=0) # sum all area chunks and terminate # transpose to allow broadcasting of multiplication

def permute_first_and_last(x:ndarray):
    shape_x = x.shape
    permute = list(arange(len(shape_x))) # order of dimension
    permute_first_last = permute[1:]+[permute[0]] # new order of dimensions with dimension 0 moved to last
    return transpose(x,permute_first_last) # array transposed with first dimension as last

def areame_mixture(x_:list[ndarray]) -> float:
    '''
    : --------------------------- ∞
    cre: Mar 2022

    web: github.com/marcodeangelis
    org: University of Liverpool
    
    MIT
    : --------------------------- ∞

    Code for the area metric of an envelope of datasets. Currenly under speed testing. Looking for more efficient implementations.
    
    ''' 
    x = mixture(x_)
    one_dataseries = dataseries(x.values[1]) # sort takes place in here. # collects all values into one dataseries.
    one_sorted_values = one_dataseries.value_sorted
    if x.dim==(1,): return areame_mixture_inneralgorithm(x,one_sorted_values) # if mixture has dimension (1,) terminates here.
    one_sorted_values_permute_first_and_last = permute_first_and_last(one_sorted_values)
    x_dim=x.dim
    areame_array = empty(x_dim)
    for j in range(prod(x_dim)): # there should be more effcient ways to do this # it computes the area metric for each 1d mixture in the mixture array.
        i = map_index_flat_to_array(x_dim)[j]
        areame_array[i] = areame_mixture_inneralgorithm(mixture_given_dimension_index(x,i),one_sorted_values_permute_first_and_last[i])
    return areame_array
