'''
-------------------------------
Created Feb 2022
github.com/marcodeangelis
University of Liverpool
GNU General Public License v3.0
-------------------------------
'''
from __future__ import annotations
from typing import Sequence, Sized, Union, Iterable, Optional, Any, Callable, Tuple

import warnings

import numpy
from numpy import (ndarray,asarray,linspace,concatenate)

from .dataseries import DataSeries, Mixture
from areametric import dataseries as ds
# from .dataseries import (is_sized,is_iterable,is_dataseries,is_mixture,dataseries,mixture)

def is_sized(x:Any) -> bool: return ds.is_sized(x)
def is_iterable(x:Any) -> bool: return ds.is_iterable(x)
def is_dataseries(x:Any) -> bool: return ds.is_dataseries(x)
def dataseries(x:Any) -> DataSeries: return ds.dataseries(x)
def is_mixture(x:Any) -> bool: return ds.is_mixture(x)
def mixture(x:Any) -> Mixture: return ds.mixture(x)
def compatible(x:DataSeries,y:DataSeries) -> bool: return x.dim == y.dim  # if this is True area metric can be computed.
# Two DataSeries are area-metric compatible or comparable if they have the same dimension. 
# Note that same dimension does not mean same sample size. 


def ecdf(x_:ndarray, w_:ndarray=None) -> ndarray:   
    x = dataseries(x_)
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    n = len(x) 
    if w_ is None: p = linspace(1/n,1,n) 
    else: p = asarray(w_,dtype=float)*linspace(1/n,1,n) # w must add up to one
    return x.value_sorted, p

    # d = dataset_parser(x)
    # if isinstance(d, Dataset): # parsing was successful
    # try: x.classname=='DataSeries'
    # except AttributeError as attribute_error: return x 
    # return x, pvalues
    # elif isinstance(d, MixtureDataset):
        # NotImplemented # implement for mixture
    # else:
        # raise TypeError('Input data structure not recognized.')

def ecdf_(x_:ndarray, w_:ndarray=None) -> ndarray:
    x = dataseries(x_)
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    n=len(x)
    if w_ is None: pval = linspace(1/n,1,n)
    else: pval = asarray(w_,dtype=float) * linspace(1/n,1,n) # w must add up to one
    x_value_sorted = x.value_sorted
    return concatenate(([x_value_sorted[0]],x_value_sorted)), concatenate(([0.],pval))

    # if isinstance(d, Dataset): # parsing was successful
        # n = len(d) # Dataset is always Sized
        # x, pvalues = d.value()[d.index()], numpy.linspace(1/n,1,n)
        
    # elif isinstance(d, MixtureDataset):
    #     NotImplemented # implement for mixture
    # else:
    #     raise TypeError('Input data structure not recognized.')

# def ecdf_(x: Union[DATASET_TYPE,Dataset]
#         ) -> Tuple[numpy.ndarray,numpy.ndarray]:
#     concat = numpy.concatenate
#     d = dataset_parser(x) # sorting of data happens in the Dataset constructor
#     if isinstance(d, Dataset): # parsing was successful
#         n = len(d) # Dataset is always Sized
#         x, pvalues = d.value()[d.index()], numpy.linspace(1/n,1,n)
#         return concat(([x[0]],x)), concat(([0.],pvalues))
#     elif isinstance(d, MixtureDataset):
#         NotImplemented # implement for mixture
#     else:
#         raise TypeError('Input data structure not recognized.')


# def pseudoinverse_(x_: Union[ndarray,DataSeries], u_: Union[ndarray,float]) -> ndarray:
#     x = dataseries(x_) # sorting of data happens in the DataSeries constructor
#     if x.__class__.__name__ == 'Mixture': 
#         print('TODO: implement ecdf for mixtures.')
#         raise ValueError
#     n=len(x)
#     x,_= ecdf_(x) # x gets shadowed
#     u = asarray(u_,dtype=float) # if cannot cast to dtype=float return the numpy error
#     one = u==1
#     if is_sized(u):
#         index = asarray(n*u,dtype=int)+1
#         if any(one): index[one] = index[one]-1
#     else:
#         if one: return x[-1] # return the highest data point
#         u = float(u) # if python can't cast 'u' to float, then return the in-built python error
#         index = int(u*n)+1
#     return x[index]

# https://en.wikipedia.org/wiki/Cumulative_distribution_function#Inverse%20distribution%20function%20(quantile%20function)

def quantile_function(x_: Union[ndarray,DataSeries], u_: Union[ndarray,float]) -> ndarray: # https://en.wikipedia.org/wiki/Quantile_function 
    x = dataseries(x_) # sorting of data happens in the DataSeries constructor
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    n=len(x) # x,_= ecdf_(x) # x gets shadowed
    x_value_sorted = x.value_sorted
    x_value_sorted_ = concatenate(([x_value_sorted[0]],x_value_sorted))
    u = asarray(u_,dtype=float) # if cannot cast to dtype=float return the numpy error
    one = u==1
    index = asarray(n*u,dtype=int)+1
    if is_sized(u):
        if any(one): index[one] = index[one]-1
    else: 
        if one: return x_value_sorted_[-1] # return the highest data point
    return x_value_sorted_[index]

def generalised_inverse(x_: ndarray) -> Callable[[ndarray]]: # `Generalised inverse distribution function` and `quantile function` have the same meaning.
    x = dataseries(x_) # Raise custom error if x_ can't be parsed
    def fun(u: ndarray): return quantile_function(x,u)
    return fun
    
# def pseudoinverse_(x: DATASET_TYPE,u: Union[float,VECTOR_TYPE]) -> Union[float, numpy.array]:
#     d = dataset_parser(x) # sorting of data happens in the Dataset constructor
#     n=len(d)
#     x,_= ecdf_(d)
#     u = numpy.asarray(u,dtype=float) # if cannot cast to dtype=float return the numpy error
#     one = u==1
#     if is_sized(u):
#         index = numpy.asarray(n*u,dtype=int)+1
#         if any(one):
#             index[one] = index[one]-1
#     else:
#         if one: # returns the highest data point
#             return x[-1]
#         u = float(u) # if python can't cast 'u' to float returns the in-built python error
#         index = int(u*n)+1
#     return x[index]


