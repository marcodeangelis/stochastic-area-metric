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
from numpy import (ndarray,asarray,linspace,concatenate,arange,transpose,prod,empty)
from numpy.matlib import repmat
from numpy import max as numpy_max

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

def ecdf_p(x_:ndarray, w_:ndarray=None) -> ndarray:   # more efficient than `ecdf` if only p values are needed
    x = dataseries(x_)
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    n = len(x) 
    if w_ is None: p = linspace(1/n,1,n) 
    else: p = asarray(w_,dtype=float)*linspace(1/n,1,n) # w must add up to one
    return p

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

def ecdf_p_(x_:ndarray, w_:ndarray=None) -> ndarray: # more efficient than `ecdf_` if only p values are needed
    x = dataseries(x_)
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    n=len(x)
    if w_ is None: pval = linspace(1/n,1,n)
    else: pval = asarray(w_,dtype=float) * linspace(1/n,1,n) # w must add up to one
    return concatenate(([0.],pval))

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

def quantile_value(x_: Union[ndarray,DataSeries], u_: Union[ndarray,float]) -> ndarray: # https://en.wikipedia.org/wiki/Quantile_function 
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

def quantile_function(x: ndarray) -> Callable[[ndarray]]: # `Generalised inverse distribution function` and `quantile function` have the same meaning.
    def fun(u: ndarray): return quantile_value(x,u)
    return fun

def inverse_quantile_algorithm(x:ndarray, q:ndarray, p:ndarray, side='left'): # core inverse-quantile algorithm for 1d-array data, i.e. data of dimension (n,).
    n=len(x) # x must be sorted, but don't sort it in here!
    if (len(x.shape)!=1): raise ValueError(f'Array shape must be ({n},) and not {x.shape}.')
    if is_sized(q):
        m = len(q)
        if (len(q.shape)!=1): raise ValueError(f'Array shape must be ({m},) and not {q.shape}.')
        qx = repmat(q,n,1).T
        xq = repmat(x,m,1)
        i = repmat(arange(n),m,1)
        b = qx > xq if side=='left' else qx >= xq # qx and xq have the same shape so comparison can take place
        return asarray([p[k] for k in numpy_max((i+1) * b, axis=1)],dtype=float)
    b = q > x if side=='left' else q >= x
    i = arange(len(x))
    return p[max((i+1) * b)]

def quantile_tensor(x:ndarray[float],i:ndarray[int],q:ndarray[float],p:ndarray[float],side='left'): 
    q = asarray(q,dtype=float)
    shapex_mut = list(x.shape) # make shape of x mutable
    rep = shapex_mut[0] # first dimension is the number of repetitions
    qns = len(q) # first dimension will be replaced with the number of q-values to compute
    shapex_mut[0] = qns # replace rep with number of q-values
    shape_o = tuple(shapex_mut) # back to immutable
    permut = list(arange(len(shape_o))) # order of dimension
    permut_rep_last = permut[1:]+[permut[0]] # new order of dimensions with dimension 0 moved to last
    shape1 = [shape_o[p] for p in permut_rep_last]
    xt = transpose(x,permut_rep_last) # array transposed with rep dimension as last
    it = transpose(i,permut_rep_last) # argsort indexes transposed with rep dimension as last
    pq_sorted_flat = empty((prod(shape_o,dtype=int),))
    for j in range(prod(shape_o[1:],dtype=int)): # loop over all elements except the first dimension
        start_x,end_x = int(j*rep), int((j+1)*rep)
        start_q,end_q = int(j*qns), int((j+1)*qns)
        x_sorted_flat = xt.flatten()[start_x:end_x][it.flatten()[start_x:end_x]] # <- sort happens here (sort algorithm not deployed here)
        pq_sorted_flat[start_q:end_q] = inverse_quantile_algorithm(x_sorted_flat,q,p,side=side) # here deploy quantile algorithm
    premut_rep_back = [permut[-1]]+permut[:-1] # permute back rep dimension to occupy the first
    pq_sorted_reshape = pq_sorted_flat.reshape(shape1) # from flat back to shape1
    pq_sorted_reshape_transpose = transpose(pq_sorted_reshape,premut_rep_back) # from shape1 back to shape0
    return pq_sorted_reshape_transpose # if values are already sorted (increasing) this must return x

def quantile_tensor_float(x:ndarray[float],i:ndarray[int],q:float,p:ndarray[float],side='left'): 
    shape_x = x.shape
    rep = shape_x[0]
    shape_o = shape_x[1:]
    permut = list(arange(len(shape_x))) # order of dimension
    permut_rep_last = permut[1:]+[permut[0]] # new order of dimensions with dimension 0 moved to last
    shape1 = [shape_x[p] for p in permut_rep_last]
    xt = transpose(x,permut_rep_last) # array transposed with rep dimension as last
    it = transpose(i,permut_rep_last) # argsort indexes transposed with rep dimension as last
    m=prod(shape_o,dtype=int)
    pq_sorted_flat = empty((m,))
    for j in range(m): # loop over all elements except the first dimension
        start_x,end_x = int(j*rep), int((j+1)*rep)
        x_sorted_flat = xt.flatten()[start_x:end_x][it.flatten()[start_x:end_x]] # <- sort happens here (sort algorithm not deployed here)
        pq_sorted_flat[j] = inverse_quantile_algorithm(x_sorted_flat,q,p,side=side) # here deploy quantile algorithm
    premut_rep_back = [permut[-1]]+permut[:-1] # permute back rep dimension to occupy the first
    pq_sorted_reshape = pq_sorted_flat.reshape(shape_o) # from flat back to shape1
    return pq_sorted_reshape # if values are already sorted (increasing) this must return x

def inverse_quantile_value(x_:Union[ndarray,DataSeries], q:Union[ndarray,float],side='left') -> Union[ndarray,float]:
    x = dataseries(x_) # sorting of data happens in the DataSeries constructor # sort algorithm is invoked here, but only if x is not a DataSeries.
    if x.__class__.__name__ == 'Mixture': 
        print('TODO: implement ecdf for mixtures.')
        raise ValueError
    p = ecdf_p_(x) # vector to be indexed by k
    if is_sized(q): return quantile_tensor(x.value,x.index,q,p,side=side) # array of indices
    else: return quantile_tensor_float(x.value,x.index,q,p,side=side) # array of indices
    
    
    # if is_sized(q): # must iterate over dataseries dimensions
    #     q = asarray(q,dtype=float)
    #     m = len(q)
    #     qx = repmat(q,n,1).T
    #     xq = repmat(x_value_sorted,m,1)
    #     i = repmat(arange(n),m,1)
    #     lo = qx > xq
    #     indices = numpy_max((i+1)*lo, axis=1)
    #     return asarray([p[k] for k in indices],dtype=float)
    # b = q > x_value_sorted
    # i=arange(len(x))
    # return p[max((i+1)*b)]

def inverse_quantile_function(x:ndarray,side='left') -> Callable[[ndarray]]:
    def fun(q:ndarray): return inverse_quantile_value(x,q,side=side)
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


def inverse_confidence_band_vector(d,u,alpha=0.05):
    n=len(d)
    h = numpy.sqrt(((numpy.log(2/alpha))/(2*n)))
    x,y=ecdf(d)
    y_hi = y+h
    y_lo = y-h
    y_hi[y_hi>1]=1
    y_lo[y_lo<0]=0
    o = int(h//(1/n)+1)
    a = h+(1/n)
    b = h+(1/n)*(n-o)
    u = numpy.asarray(u,dtype=float)
    index_le =  numpy.asarray((u-a)//(1/n) + 1, dtype=int) # numpy.empty(u.shape)
    index_le[u<a]=0
    index_le[u>b]=-1
    index_ri = - numpy.asarray(((1-u)-a)//(1/n) + 2 , dtype=int)
    index_ri[(1-u)<a]=-1
    index_ri[(1-u)>b]=0
    x_left  = numpy.asarray([-2]+list(x[1:-o+1]))
    x_right = numpy.asarray(list(x[o-1:-1])+[10])
    left = numpy.asarray([x_left[i] for i in index_le])
    right= numpy.asarray([x_right[i] for i in index_ri])
    return y_hi,y_lo, x, left, right


def inverse_confidence_band(d,u,alpha=0.05):
    n=len(d)
    h = numpy.sqrt(((numpy.log(2/alpha))/(2*n)))
    x,y=ecdf(d)
    y_hi = y+h
    y_lo = y-h
    y_hi[y_hi>1]=1
    y_lo[y_lo<0]=0
    o = int(h//(1/n)+1)
    a = h+(1/n)
    b = h+(1/n)*(n-o)
    if u<a:
        index_le=0
    elif u>b:
        index_le=-1
    else:
        index_le = int( (u-a)//(1/n) + 1 )
    x_left = numpy.asarray([-2]+list(x[1:-o+1]))
    if (1-u)<a:
        index_ri=-1
    elif (1-u)>b:
        index_ri=0
    else:
        index_ri = -int( ((1-u)-a)//(1/n) + 2 )
    x_right = numpy.asarray(list(x[o-1:-1])+[10])
    return y_hi,y_lo, x, x_left[index_le],x_right[index_ri]