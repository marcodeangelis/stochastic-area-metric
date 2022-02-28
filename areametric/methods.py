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

def parser(x_:Any): # parse any possible data structure into a DataSeries
    if x_.__class__.__name__ == 'DataSeries': return x_
    if x_.__class__.__name__ == 'Mixture': return x_
    try: x=asarray(x_,dtype=float)
    except ValueError as value_error: 
        print('Data does not qualify to be a DataSeries.\n Try to use `mixture(x)` instead.') # ValueError: setting an array element with a sequence.
        return x_
    return DataSeries(x)




def compatible(x:DataSeries,y:DataSeries): return x.dim == y.dim  # if this is True area metric can be computed.
# Two DataSeries are area-metric compatible or comparable if they have the same dimension. 
# Note that same dimension does not mean same sample size. 

# def dataset_parser(x: Union[DATASET_TYPE,Sequence[Sequence[float]]]
#                   ) -> Union[Dataset, MixtureDataset]:
#     if isinstance(x,Dataset):
#         return x # no need to parse if x is already a Dataset
#     try: x_arr = numpy.asarray(x, dtype=float) # numeric types
#     except: x_arr = numpy.asarray(x) # mixtures # convert data structure to numpy # piggyback on the powerful asarray parser
#     x_shape = x_arr.shape
#     is_interval = False
#     if x_arr.dtype.name == 'object': # mixture dataset
#         return MixtureDataset(x, homogeneous = False)
#     elif x_arr.dtype.name in NUMBERS: # literal dataset, complex numbers won't be allowed
#         if len(x_shape)==0: # single value dataset
#             return Dataset([x_arr])
#         if len(x_shape)==1: # single column/row dataset
#             return Dataset(x_arr)
#         elif len(x_shape)==2: # mixture dataset # nested sequences have all the same size
#             for s in x_shape:
#                 if s==2: is_interval = True # possible interval dataset, this could be intervalized
#             return MixtureDataset(x) # interval = is_interval
#         elif len(x_shape)>2:
#             warnings.warn('Input cannot be a tensor with dim>2. Input will be returned.')
#     else:
#         warnings.warn('Input not recognized. Input must be a Sequence or a nested Sequence of floats. Input will be returned.')
#     return x

def ecdf(x_:ndarray, w_:ndarray=None) -> ndarray:   
    x,w=parser(x_), asarray(w_,dtype=float) # if this is successful carry on
    n = len(x) 
    if w is None: return x.value[x.index], linspace(1/n,1,n) 
    else: return x.value[x.index], w*linspace(1/n,1,n)  # w must add up to one

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
    x,w = parser(x_), asarray(w_,dtype=float)
    n=len(x)
    if w is None: xval, pval = x.value[x.index], linspace(1/n,1,n)
    else: xval, pval = x.value[x.index], w*linspace(1/n,1,n)  # w must add up to one
    return concatenate(([xval[0]],xval)), concatenate(([0.],pval))

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


def pseudoinverse_(x_: ndarray,u: ndarray) -> ndarray:
    x = parser(x_) # sorting of data happens in the DataSeries constructor
    n=len(x)
    x,_= ecdf_(x)
    u = asarray(u,dtype=float) # if cannot cast to dtype=float return the numpy error
    one = u==1
    if sized(u):
        index = asarray(n*u,dtype=int)+1
        if any(one): index[one] = index[one]-1
    else:
        if one: # returns the highest data point
            return x[-1]
        u = float(u) # if python can't cast 'u' to float, then return the in-built python error
        index = int(u*n)+1
    return x[index]


def pseudoinverse(x_: ndarray) -> Callable[[ndarray]]:
    x = parser(x_) # Raise custom error if x_ can't be parsed
    def fun(u: ndarray): return pseudoinverse_(x,u)
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

def sized(x:Any):
    try: len(x)
    except TypeError: return False
    return True
