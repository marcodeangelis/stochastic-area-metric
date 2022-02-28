'''
-------------------------------
Created Feb 2022
github.com/marcodeangelis
University of Liverpool
GNU General Public License v3.0
-------------------------------

DataSeries and Mixture are two classes used by the areametric library. 

'''
from __future__ import annotations
from typing import Sequence, Sized, Union, Iterable, Optional, Any, Callable, Tuple

import warnings

import numpy
from numpy import (ndarray,asarray,transpose,argsort,concatenate)

MACHINE_EPS = 7./3 - 4./3 - 1

NUMBERS  =          {'int','float','complex',                   # Python numbers     
                    'int8','int16','int32','int64','intp',      # Numpy integers
                    'uint8','uint16','uint32','uint64','uintp', # Numpy unsigned integers
                    'float16','float32','float64','float_'}     # Numpy floats and doubles


def is_dataseries(x:Any) -> bool:
    if x.__class__.__name__ == 'DataSeries': return True
    try: asarray(x,dtype=float)
    except ValueError: return False 
    return True

def dataseries_value(x:Any) -> ndarray: # return ndarray if it qualifies to be a DataSeries
    if x.__class__.__name__ == 'DataSeries': return x.value
    return asarray(x,dtype=float) # cast array to float # https://realpython.com/python-exceptions/
    # if x_.__class__.__name__ != 'ndarray': raise ValueError('The provided data cannot be parsed as a DataSeries.') 

def dataseries(x:Any) -> DataSeries: # return a DataSeries if x qualifies
    if x.__class__.__name__ == 'DataSeries': return x
    try: return DataSeries(asarray(x,dtype=float)) # cast array to float # https://realpython.com/python-exceptions/
    except ValueError: return x

def is_sized(x:Any) -> bool:
    try: len(x)
    except TypeError: return False # TypeError: object of type ... has no len()
    return True

def is_iterable(x:Any) -> bool:
    try: iter(x)
    except TypeError: return False # TypeError: ... object is not iterable
    return True

def compatible(x:DataSeries,y:DataSeries) -> bool: return x.dim == y.dim  # if this is True, area metric can be computed.

def show(x:DataSeries, n:int=10) -> str:
    len_x = len(x)
    if len_x > 2*n: 
        a,b = [f'{xi}' for xi in x[:n]],[f'{xi}' for xi in x[-n:]]
        return '\n'.join(a+['...']+b)
    else: return f'{x.value}'
    # else: return '\n'.join([f'{xi}' for xi in x])

#  print('The data does not qualify to be a DataSeries because it is composed of arrays of heterogeneous size.\n To parse the data, use Mixture instead.')
# ('Rows are longer than columns: the value array will be transposed.\nIf there are more dimensions than repetitions, set the flag `shallow=True`.')
# if len(x.shape)>2: raise ValueError(f'Unexpected shape of input ndarray: {x.shape} was provided, while (n,d) was expected.') # Tabular data can be at most a 2d-array
class DataSeries(object):
    """
    -------------------------------
    Created Feb 2022
    github.com/marcodeangelis
    University of Liverpool
    MIT - License
    -------------------------------

    DataSeries is a wrapper of the Numpy ndarray class. 

    (*) The value of a DataSeries is a 1d-ndarray or a Nd-array
    (*) Arithmetic between DataSeries can be accessed through their value
    (*) A DataSeries is (1) sized, (2) iterable, and (3) indexable. 
    (*) A DataSeries is not hashable
    (*) Two compatible DS x and y can be concatenated with x+y

    When instantiated the DataSeries constructor computes and stores the indices of the sorted array (across the first dimension).

    The first dimension signifies the sample size or number of repetitions.

    A DS which value is a 2d-array is a tabular DataSeries, which is someway in between a DS and a Mixture. 
    Such tabular DS can be seen as a Mixture with columns of homogeneous size. 

    """
    def __repr__(self): return show(self)# return
    def __str__(self): return show(self) # print
    def __init__(self,x_: ndarray, dtype=float, shallow:bool=True) -> None: # constructor must return None
        self.__value = dataseries_value(x_) # turn x into ndarray if it qualifies to be a DataSeries
        # if x_.__class__.__name__ == 'DataSeries': self.__value = x.value 
        # else: self.__value = x
        self.__tabular = True
        if len(self.__value.shape)==1: self.__tabular = False
        elif len(self.__value.shape)==2: 
            if (self.__value.shape[1]>self.__value.shape[0]): 
                if shallow==False: warnings.warn('There are more dimensions than repetitions. Set the flag `shallow=True` to automatically transpose the value array.')
                if shallow==True: 
                    warnings.warn('There are more dimensions than repetitions. The flag `shallow` is set to True so, the value array will be automatically transposed.')
                    self.__value = transpose(self.__value)
        self.__shape = self.__value.shape
        self.__index = argsort(self.__value,axis=0)
        if len(self.__shape)==1: self.__dim = 1
        else: self.__dim = tuple([self.__shape[j] for j in range(1,len(self.__shape))])
        self.__length = self.__shape[0]
    def __len__(self) -> int: return self.__shape[0]
    def __iter__(self): # makes class iterable
        for v in self.__value: yield v 
    def __next__(self): pass 
    def __getitem__(self, i:Union[int,slice]): return self.__value[i]
    def __setitem__(self, i: Union[int, slice], x: Union[ndarray,float]) -> None: self.__value[i] = x
    # -------------- PROPERTY METHODS -------------- #
    @property
    def value(self): return self.__value
    @property
    def index(self): return self.__index
    @property
    def shape(self): return self.__shape
    @property
    def tabular(self): return self.__tabular
    @property
    def dim(self): return self.__dim
    @property
    def length(self): return self.__length
    @property
    def repetitions(self): return self.__length
    @property
    def classname(self): return self.__class__.__name__
    @property
    def info(self):
        d={ 'class':self.classname,
            'rep':self.repetitions,
            'dim':self.dim,
            'tabular':self.tabular,}
        return d
    # -------------- MAGIC METHODS ----------------- #
    def __add__(self,other) -> DataSeries: # for concatenation
        otherType = other.__class__.__name__
        if otherType == 'DataSeries': return DataSeries(concatenate((self.value,other.value),axis=0))
        elif otherType == 'Mixture': return Mixture([self.value] + other.value)
        else: x = mixture(other)
        return Mixture([self.value] + x.value)
    def __radd__(left,self): return self.__add__(self,left)
    def __mul__(self, c:int) -> DataSeries: # c must be integer
        r=self
        for _ in range(1,c): r+=self
        return r
    def __rmul__(self,left: int) -> DataSeries:
        return self*left

def dimension(shapes:list) -> Tuple: # establish the dimension of the mixture (it could be multiple).
    s_iter = iter(shapes)
    si = next(s_iter)
    if len(si)==1: return (1,)
    else: return tuple([si[j] for j in range(1,len(si))])

def is_mixture(x:Any) -> bool:
    if x.__class__.__name__ == 'Mixture': return True
    if x.__class__.__name__ == 'DataSeries': return True
    if is_sized(x)==False: return False
    if is_iterable(x)==False: return False
    if is_dataseries(x): return True # a tabular DataSeries is the simplest mixture
    return all([is_dataseries(xi) for xi in x])

def mixture_value(x:Any) -> Sequence[ndarray]: # return the mixture value
    if x.__class__.__name__ == 'Mixture': return x.value
    if x.__class__.__name__ == 'DataSeries': return [x.value]
    if is_iterable(x)==False: raise ValueError('Input does not qualify to be a dataseries mixture.')
    if is_mixture(x): mix = [dataseries(xi) for xi in x]
    else: raise ValueError('Input does not qualify to be a dataseries mixture.')
    if all([compatible(xi,xj) for xi in mix for xj in mix]): return mix
    else: raise ValueError('Input does not qualify to be a mixture because dataseries have heterogeneous dimensions.')

def mixture(x:Any) -> Mixture: # return the Mixture object # this funtion should be used to parse data into Mixture rather than the constructor.
    if x.__class__.__name__ == 'Mixture': return x
    if x.__class__.__name__ == 'DataSeries': return Mixture(x)
    if is_iterable(x)==False: raise ValueError('Input does not qualify to be a dataseries mixture.')
    if is_mixture(x): m = [dataseries(xi) for xi in x]
    else: raise ValueError('Input does not qualify to be a dataseries mixture.')
    if all([compatible(xi,xj) for xi in m for xj in m]): return Mixture(m)
    else: raise ValueError('Input does not qualify to be a mixture because dataseries have heterogeneous dimensions.')

class Mixture(object):
    """
    -------------------------------
    Created Feb 2022
    github.com/marcodeangelis
    University of Liverpool
    MIT - License
    -------------------------------

    Mixture is a collection of DataSeries objects. It is an abstraction of DataSeries to include data with heterogeneous size. 
    Here size refers to the number of repetitions, i.e. the first integer in the shape of each DataSeries.

    Not all data structures can be parsed into a Mixture. 

    (*) The data structure must be a sequence of array-like structures
    (*) Each array-like structure must be a compatible DataSeries

    Two DataSeries are compatible if their shape differs only in the size of the first dimension.
    For example, if x.shape is (100,2,3) and y.shape is (20,2,3) then, x and y are compatible. 
    If we say that x and y have dimension (2,3) then, two DataSeries are compatible if they have the same dimension.
    The first dimension of a DataSeries is the number of repetitions.

    Mixture collates data with different number of repetitions and with the same dimension. 

    If the mixture has all samples with the same number of repetitions, it is called a homogeneous mixture.

    """
    def __repr__(self) -> str: return f'{self.value}'
    def __str__(self) -> str: return f'{self.value}'
    def __init__(self,x_:Sequence[DataSeries,ndarray]):
        # x =  # return an iterable of compatible DataSeries, i.e. DataSeries with the same dimension 
        # if x_.__class__.__name__ == 'Mixture': self.__value = x_.value
        # else: self.__value = mixture_(x_)
        self.__value = mixture_value(x_)
        self.__shapes = [xi.shape for xi in self.__value]
        self.__lengths = [len(xi) for xi in self.__value]
        self.__dim = dimension(self.__shapes)
        self.__homogeneous = False
        if all([len(xi)==len(xj) for xi in self.__value for xj in self.__value]): self.__homogeneous = True
    def __len__(self): return len(self.value)
    def __iter__(self): # makes class iterable
        for v in self.__value: yield v 
    def __next__(self): pass 
    def __getitem__(self, i:Union[int,slice]): return dataseries(self.__value[i])
    def __setitem__(self, i: Union[int, slice], x: Union[ndarray,float]) -> None: self.__value[i] = x
    @property
    def shapes(self): return self.__shapes
    @property
    def lengths(self): return self.__lengths
    @property
    def repetitions(self): return self.__lengths
    @property
    def dim(self): return self.__dim
    @property
    def homogeneous(self): return self.__homogeneous
    @property
    def value(self): return self.__value
    @property
    def classname(self): return self.__class__.__name__
    @property
    def info(self):
        d={ 'class':self.classname,
            'rep':self.lengths,
            'dim':self.dim,
            'len':len(self.value),
            'hom':self.homogeneous}
        return d
    # -------------- MAGIC METHODS ----------------- #
    def __add__(self,other) -> Mixture: # for concatenation
        otherType = other.__class__.__name__
        if otherType == 'Mixture': return Mixture(self.value + other.value)
        elif otherType == 'DataSeries': return Mixture(self.value + [other.value])
        else: x = mixture(other) # will raise error if it does not qualify
        return self.__add__(self,x)
    def __radd__(left,self):
        leftType = left.__class__.__name__
        if leftType == 'DataSeries': return self.__add__(self,left)
        else: x = mixture(left)
        return self.__add__(self,x)

# class Dataset(): # mutable sequence of numbers
#     '''
#     A dataset class. Mutable, some arithmetic behaviour is list like.

#     cre: Mon Jan 4 18:33 2020 
#     edi: Mon Jun 14 22:05 2021
#     @author: Marco De Angelis 
#     '''
#     def __repr__(self): # return
#         return show(self)
#     def __str__(self): # print
#         return show(self)
#     def __len__(self):
#         return len([i for i in self])
#     def __iter__(self): # makes class iterable
#         for v in self.__value: yield v
#     def __next__(self):
#         pass 
#     def __getitem__(self, index: Union[int, slice, Sequence[int]]): # make class indexable
#         if isinstance(index,int):
#             return self.__value[index]
#         elif isinstance(index,slice): # this should be the fastest
#             return self.__value[index]
#         elif isinstance(index, Sequence):
#             return numpy.asarray([self.__value[i] for i in index], dtype=float)
#     def __setitem__(self, index: Union[int, slice], x: float): 
#         self.__value[index] = x
#     def __init__(self,x: numpy.ndarray):
#         self.__value = x
#         self.__index = numpy.argsort(x)
#     def value(self):
#         return self.__value
#     def index(self):
#         return self.__index
#     def __add__(self,other):
#         o = dataset_parser(other)
#         if isinstance(o,Dataset):
#             return Dataset(numpy.concatenate((self.value(),o.value())))
#         return self
#     def __mul__(self,c: int) -> Dataset: # c must be integer
#         if isinstance(c,int):
#             self_copy = self
#             for i in range(1,c):
#                 self_copy += self
#             return self_copy
#         return self
#     def __rmul__(self,left: int) -> Dataset:
#         if isinstance(left,int):
#             self_copy = self
#             for i in range(left):
#                 self_copy += self
#             return self_copy
#         return self
#     def __eq__(self,other):
#         for s,o in zip(self,other):
#             if s!=o:
#                 return False
#         return True

# class MixtureDataset(): # just an iterable of Datasets, underdeveloped
#     def __init__(self,x: numpy.ndarray, homogeneous: bool=False):
#         self.__value = []
#         for xi in x:
#             self.__value.append(numpy.asarray(xi,dtype=float))
#     def value(self):
#         return self.__value



