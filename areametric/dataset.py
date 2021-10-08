'''
    %---------------------------------------#
    Tue June 22 17:15 2021
    @author: Marco De Angelis 
    github.com/marcodeangelis
    GNU General Public License v3.0
    %---------------------------------------#

    A module for the dataset class.

'''
from __future__ import annotations

from typing import Sequence, Sized, Union, Iterable, Optional, Any, Callable, Tuple

import warnings

import numpy

MACHINE_EPS = 7./3 - 4./3 - 1

NUMBERS  =          {'int','float','complex',                   # Python numbers     
                    'int8','int16','int32','int64','intp',      # Numpy integers
                    'uint8','uint16','uint32','uint64','uintp', # Numpy unsigned integers
                    'float16','float32','float64','float_'} #,  # Numpy floats and doubles

INTEGERS =          {'int','int8','int16','int32','int64','intp','uint8','uint16','uint32','uint64','uintp'}
FLOATS =            {'float','float16','float32','float64','float_'}

DATASET_TYPE = Union[float,int,Sequence[float],numpy.ndarray]
VECTOR_TYPE = Union[numpy.ndarray,Sequence[float]]

def show(x: Iterable, N: int=5) -> str:
    if len(x)>10:
        a1,a2 = [str(i) for i in x[:N]], [str(i) for i in x[-N:]]
        return '\n'.join(a1+['...']+a2)
    else: return '\n'.join([str(i) for i in x])

class Dataset(): # mutable sequence of numbers
    '''
    A dataset class. Mutable, some arithmetic behaviour is list like.

    cre: Mon Jan 4 18:33 2020 
    edi: Mon Jun 14 22:05 2021
    @author: Marco De Angelis 
    '''
    def __repr__(self): # return
        return show(self)
    def __str__(self): # print
        return show(self)
    def __len__(self):
        return len([i for i in self])
    def __iter__(self): # makes class iterable
        for v in self.__value: yield v
    def __next__(self):
        pass 
    def __getitem__(self, index: Union[int, slice, Sequence[int]]): # make class indexable
        if isinstance(index,int):
            return self.__value[index]
        elif isinstance(index,slice): # this should be the fastest
            return self.__value[index]
        elif isinstance(index, Sequence):
            return numpy.asarray([self.__value[i] for i in index], dtype=float)
    def __setitem__(self, index: Union[int, slice], x: float): 
        self.__value[index] = x
    def __init__(self,x: numpy.ndarray):
        self.__value = x
        self.__index = numpy.argsort(x)
    def value(self):
        return self.__value
    def index(self):
        return self.__index
    def __add__(self,other):
        o = dataset_parser(other)
        if isinstance(o,Dataset):
            return Dataset(numpy.concatenate((self.value(),o.value())))
        return self
    def __mul__(self,c: int) -> Dataset: # c must be integer
        if isinstance(c,int):
            self_copy = self
            for i in range(1,c):
                self_copy += self
            return self_copy
        return self
    def __rmul__(self,left: int) -> Dataset:
        if isinstance(left,int):
            self_copy = self
            for i in range(left):
                self_copy += self
            return self_copy
        return self
    def __eq__(self,other):
        for s,o in zip(self,other):
            if s!=o:
                return False
        return True

class MixtureDataset(): # just an iterable of Datasets, underdeveloped
    def __init__(self,x: numpy.ndarray, homogeneous: bool=False):
        self.__value = []
        for xi in x:
            self.__value.append(numpy.asarray(xi,dtype=float))
    def value(self):
        return self.__value

def dataset_parser(x: Union[DATASET_TYPE,Sequence[Sequence[float]]]
                  ) -> Union[Dataset, MixtureDataset]:
    if isinstance(x,Dataset):
        return x # no need to parse if x is already a Dataset
    try: x_arr = numpy.asarray(x, dtype=float) # numeric types
    except: x_arr = numpy.asarray(x) # mixtures # convert data structure to numpy # piggyback on the powerful asarray parser
    x_shape = x_arr.shape
    is_interval = False
    if x_arr.dtype.name == 'object': # mixture dataset
        return MixtureDataset(x, homogeneous = False)
    elif x_arr.dtype.name in NUMBERS: # literal dataset, complex numbers won't be allowed
        if len(x_shape)==0: # single value dataset
            return Dataset([x_arr])
        if len(x_shape)==1: # single column/row dataset
            return Dataset(x_arr)
        elif len(x_shape)==2: # mixture dataset # nested sequences have all the same size
            for s in x_shape:
                if s==2: is_interval = True # possible interval dataset, this could be intervalized
            return MixtureDataset(x) # interval = is_interval
        elif len(x_shape)>2:
            warnings.warn('Input cannot be a tensor with dim>2. Input will be returned.')
    else:
        warnings.warn('Input not recognized. Input must be a Sequence or a nested Sequence of floats. Input will be returned.')
    return x

def ecdf(x: Union[DATASET_TYPE,Dataset]
        ) -> Tuple[numpy.ndarray,numpy.ndarray]:
    d = dataset_parser(x)
    if isinstance(d, Dataset): # parsing was successful
        n = len(d) # Dataset is always Sized
        x, pvalues = d.value()[d.index()], numpy.linspace(1/n,1,n)
        return x, pvalues
    elif isinstance(d, MixtureDataset):
        NotImplemented # implement for mixture
    else:
        raise TypeError('Input data structure not recognized.')

def ecdf_(x: Union[DATASET_TYPE,Dataset]
        ) -> Tuple[numpy.ndarray,numpy.ndarray]:
    concat = numpy.concatenate
    d = dataset_parser(x) # sorting of data happens in the Dataset constructor
    if isinstance(d, Dataset): # parsing was successful
        n = len(d) # Dataset is always Sized
        x, pvalues = d.value()[d.index()], numpy.linspace(1/n,1,n)
        return concat(([x[0]],x)), concat(([0.],pvalues))
    elif isinstance(d, MixtureDataset):
        NotImplemented # implement for mixture
    else:
        raise TypeError('Input data structure not recognized.')

def pseudoinverse(d: DATASET_TYPE) -> Callable[[float,numpy.array]]:
    if isinstance(d, Dataset): # parsing was successful
        def fun(u: Union[float,VECTOR_TYPE]):
            return pseudoinverse_(d,u)
    return fun
    
def pseudoinverse_(x: DATASET_TYPE,u: Union[float,VECTOR_TYPE]) -> Union[float, numpy.array]:
    d = dataset_parser(x) # sorting of data happens in the Dataset constructor
    n=len(d)
    x,_= ecdf_(d)
    one = u==1
    if is_sized(u):
        u = numpy.asarray(u,dtype=float) # if cannot cast to dtype=float return the numpy error
        index = numpy.asarray(n*u,dtype=int)+1
        if any(one):
            index[one] = index[one]-1
    else:
        if one: # returns the highest data point
            return x[-1]
        u = float(u) # if python can't cast 'u' to float returns the in-built python error
        index = int(u*n)+1
    return x[index]

def is_sized(x:Any):
    try: len(x)
    except: return False
    return True


