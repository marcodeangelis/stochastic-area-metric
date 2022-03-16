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

import itertools

import warnings

import numpy
from numpy import (ndarray,asarray,transpose,argsort,concatenate,arange,prod,empty)

# MACHINE_EPS = 7./3 - 4./3 - 1

# NUMBERS  =          {'int','float','complex',                   # Python numbers     
#                     'int8','int16','int32','int64','intp',      # Numpy integers
#                     'uint8','uint16','uint32','uint64','uintp', # Numpy unsigned integers
#                     'float16','float32','float64','float_'}     # Numpy floats and doubles

INTEGERS = {'int','int8','int16','int32','int64','intp','uint8','uint16','uint32','uint64','uintp'}

def is_dataseries(x:Any) -> bool:
    if x.__class__.__name__ == 'DataSeries': return True
    try: asarray(x,dtype=float)
    except ValueError: return False 
    return True

def dataseries_value(x:Any) -> ndarray: # return ndarray if it qualifies to be a DataSeries
    if x.__class__.__name__ == 'DataSeries': return x.value
    return asarray(x,dtype=float) # cast array to float # https://realpython.com/python-exceptions/ # it will raise Numpy exception if 
    # if x_.__class__.__name__ != 'ndarray': raise ValueError('The provided data cannot be parsed as a DataSeries.') 

def dataseries(x:Any) -> DataSeries: # return a DataSeries if x qualifies
    if x.__class__.__name__ == 'DataSeries': return x
    if x.__class__.__name__ == 'ndarray': return DataSeries(x)
    if is_dataseries(x): return DataSeries(asarray(x,dtype=float))
    if is_mixture(x): 
        warnings.warn('Input qualifies to be a mixture and it will be parsed as such.\n A mixture is a sequence of DataSeries with the same dimension, but different number of repetitions.')
        return mixture(x) 
    else: raise ValueError('The input cannot be parsed into a DataSeries nor into a mixture of DataSeries. Most likely because the inputted DataSeries have different dimension.')

def is_sized(x:Any) -> bool:
    try: len(x)
    except TypeError: return False # TypeError: object of type ... has no len()
    return True

def is_iterable(x:Any) -> bool:
    try: iter(x)
    except TypeError: return False # TypeError: ... object is not iterable
    return True

def is_integer(x:Any): 
    if x.__class__.__name__ in INTEGERS: return True
    else: return False

def compatible(x:DataSeries,y:DataSeries) -> bool: return x.dim == y.dim  # if this is True, area metric can be computed.

def iterate_with_flattened_dimension(x_:DataSeries): 
    if x_.__class__.__name__ == 'DataSeries': x=x_
    elif is_dataseries(x_): x=DataSeries(asarray(x,dtype=float))
    else: raise ValueError('Provided input is not a DataSeries.')
    xx = asarray([x.value[i].flatten(order='C') for i in range(len(x))], dtype=float).T
    try:
        for xi in xx: yield xi
    except StopIteration: pass
        
def samples_with_flattened_dimension(x_:DataSeries) -> ndarray: 
    if x_.__class__.__name__ == 'DataSeries': x=x_
    elif is_dataseries(x_): x=DataSeries(asarray(x,dtype=float))
    else: raise ValueError('Provided input is not a DataSeries.')
    xx = asarray([x.value[i].flatten(order='C') for i in range(len(x))], dtype=float).T
    return asarray([xi for xi in xx],dtype=float)

def map_c_order(i:tuple[int],dim:tuple[int]) -> ndarray: return arange(prod(dim),dtype=int).reshape(dim)[i] # return array of indices corresponding to the flattened array 
    
def samples_given_dimension(x_:DataSeries,i:tuple[int]):
    if x_.__class__.__name__ == 'DataSeries': x=x_
    elif is_dataseries(x_): x=DataSeries(asarray(x,dtype=float))
    else: raise ValueError('Provided input is not a DataSeries.')
    j = map_c_order(i,x.dim) # index of flattened array 
    if is_integer(j) | isinstance(j,slice): return samples_with_flattened_dimension(x)[j]
    else: return asarray([samples_with_flattened_dimension(x)[jj] for jj in j],dtype=float)

def map_index_flat_to_array(dim): return [t for t in itertools.product(*[arange(di) for di in dim])]

def value_sorted(x:ndarray[float],i:ndarray[int]): # return the ndarray `x` with the first dimension sorted (increasing) according to the indexes `i`
    shape0 = x.shape
    rep = shape0[0] # first dimension is the number of repetitions, i.e. the dimension to be sorted
    permut = list(arange(len(shape0))) # order of dimension
    permut_rep_last = permut[1:]+[permut[0]] # new order of dimensions with dimension 0 moved to last
    shape1 = [shape0[p] for p in permut_rep_last]
    xt = transpose(x,permut_rep_last) # array transposed with rep dimension as last
    it = transpose(i,permut_rep_last) # argsort indexes transposed with rep dimension as last
    x_sorted_flat = empty((prod(shape0,dtype=int),))
    xt_flatten,it_flatten = xt.flatten(),it.flatten()
    for j in range(prod(shape0[1:],dtype=int)): # loop over all elements except the rep dimension
        start,end = int(j*rep), int((j+1)*rep)
        x_sorted_flat[start:end]=xt_flatten[start:end][it_flatten[start:end]] # <- sort happens here (sort algorithm not deployed here)
    premut_rep_back = [permut[-1]]+permut[:-1] # permute back rep dimension to occupy the first
    x_sorted_reshape = x_sorted_flat.reshape(shape1) # from flat back to shape1
    x_sorted_reshape_transpose = transpose(x_sorted_reshape,premut_rep_back) # from shape1 back to shape0
    return x_sorted_reshape_transpose # if values are already sorted (increasing) this must return x

def show(x:DataSeries, n:int=10) -> str:
    len_x = len(x)
    xv = x.value 
    if len_x > 2*n: 
        a,b = [f'{xi}' for xi in xv[:n]],[f'{xi}' for xi in xv[-n:]]
        return '\n'.join(a+['...']+b)
    else: return f'{x.value}'
    # else: return '\n'.join([f'{xi}' for xi in x])

#  print('The data does not qualify to be a DataSeries because it is composed of arrays of heterogeneous size.\n To parse the data, use Mixture instead.')
# ('Rows are longer than columns: the value array will be transposed.\nIf there are more dimensions than repetitions, set the flag `shallow=True`.')
# if len(x.shape)>2: raise ValueError(f'Unexpected shape of input ndarray: {x.shape} was provided, while (n,d) was expected.') # Tabular data can be at most a 2d-array
class DataSeries(object):
    """ 
    :+++++++++++++++++++++++++++++++
     created: Feb 2022
     web: github.com/marcodeangelis
     org: University of Liverpool

     MIT - License
    :+++++++++++++++++++++++++++++++

    DataSeries is a wrapper of the Numpy ndarray class. 

    (*) The value of a DataSeries is a 1d-ndarray or a Nd-array
    (*) Arithmetic between DataSeries can be accessed through their value
    (*) A DataSeries is (1) sized, (2) iterable, and (3) indexable. 
    (*) A DataSeries is not hashable.
    (*) Two compatible DS x and y can be concatenated with x+y.

    When instantiated, the DataSeries constructor computes and stores the indices of the sorted array (across the first dimension).
    In other words, the DataSeries constructor deploy the sorting algorithm at inception.

    The first dimension signifies the sample size or number of repetitions the data have.

    A DS which value is a Nd-array is a tabular DataSeries, which is someway in between a DS and a Mixture. 
    Such tabular DS can be seen as a Mixture with columns of homogeneous size. 

    """
    def __repr__(self): return show(self)# return
    def __str__(self): return show(self) # print
    def __init__(self,x_: ndarray, dtype=float, shallow:bool=False, labels:Union[int,str]=None) -> None: # constructor must return None
        self.__value = dataseries_value(x_) # turn x into ndarray if it qualifies to be a DataSeries
        self.__tabular = True
        if len(self.__value.shape)==1: self.__tabular = False
        elif len(self.__value.shape)==2: 
            if (self.__value.shape[1]>self.__value.shape[0]): 
                if shallow==False: warnings.warn('There are more dimensions than repetitions. Set the flag `shallow=True` to automatically transpose the value array.')
                if shallow==True: 
                    warnings.warn('There are more dimensions than repetitions. The flag `shallow` is set to True so, the value array will be automatically transposed.')
                    self.__value = transpose(self.__value)
        self.__shape = self.__value.shape
        self.__index = argsort(self.__value,axis=0) # perform sort
        if len(self.__shape)==1: self.__dim = (1,)
        else: self.__dim = tuple([self.__shape[j] for j in range(1,len(self.__shape))])
        self.__length = self.__shape[0]
        self.__labels = labels
    def __len__(self) -> int: return self.__shape[0]
    def __iter__(self): # makes class iterable
        for v in self.__value: yield v 
    def __next__(self): pass 
    def __getitem__(self, i:Union[int,slice]): return samples_given_dimension(self,i) #return self.__value[i]
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
    @property
    def samples(self): return samples_with_flattened_dimension(self)
    def samples_index(self): return asarray([self.index[i].flatten(order='C') for i in range(len(self))], dtype=int).T
    @property
    def samples_sorted(self): return asarray([s[i] for s,i in zip(self.samples,self.samples_index())],dtype=float)
    @property
    def value_sorted(self): return value_sorted(self.value,self.index)
    @property
    def labels(self): return self.__labels
    def iterate_samples_flat(self): pass
    def iterate_samples_flat_sorted(self): pass
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
    if all([is_dataseries(xi) for xi in x]): 
        mix = [dataseries(xi) for xi in x]
        if all([compatible(xi,xj) for xi in mix for xj in mix]): return True

def mixture_value(x:Any) -> Sequence[ndarray]: # return the mixture value
    if x.__class__.__name__ == 'Mixture': return x.value
    if x.__class__.__name__ == 'DataSeries': return [x.value]
    if is_iterable(x)==False: raise ValueError('Input does not qualify to be a mixture, because the input is not iterable.')
    if is_mixture(x): mix = [dataseries(xi) for xi in x]
    else: raise ValueError('Input does not qualify to be a mixture. This is likely due to dimension incompatibility.')
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

def mixture_to_one_dataseries_value(x_:Mixture) -> DataSeries:  
    x = mixture(x_)  
    ii = []
    x_lengths = x.repetitions
    for i,xi in enumerate(x):  
        if i==0: xv=xi.value  # todo: empty mixture  
        else: xv=concatenate((xv,xi.value))    # merge all data in one vector dataseries  
        ii+=x_lengths[i]*[i] # create corresponding vector of index provenance 
    return asarray(ii,dtype=int),xv

def mixture_given_dimension_index(x_:Mixture,i:Union[tuple[int],slice]): 
    x = mixture(x_)
    return mixture([dataseries(ds[i]) for ds in x])


class Mixture(object):
    """
    :+++++++++++++++++++++++++++++++
     created: Feb 2022
     web: github.com/marcodeangelis
     org: University of Liverpool

     MIT - License
    :+++++++++++++++++++++++++++++++

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
    def __repr__(self) -> str: return f'{self.dataseries}'
    def __str__(self) -> str: return f'{self.dataseries}'
    def __init__(self,x_:Sequence[DataSeries,ndarray]):
        self.__dataseries = mixture_value(x_) # list of dataseries
        self.__shapes = [xi.shape for xi in self.__dataseries]
        self.__lengths = [len(xi) for xi in self.__dataseries]
        self.__dim = dimension(self.__shapes)
        self.__homogeneous = False
        if all([len(xi)==len(xj) for xi in self.__dataseries for xj in self.__dataseries]): self.__homogeneous = True
        # self.__values, self.__indexes = mixture_to_one_dataseries_value(self) # chain all dataseries into one <- this is needed for computing the area metric of its envelope
    def __len__(self): return len(self.__dataseries)
    def __iter__(self): # makes class iterable
        for v in self.__dataseries: yield v 
    def __next__(self): pass 
    def __getitem__(self, i:Union[int,slice]): return dataseries(self.__dataseries[i])
    def __setitem__(self, i: Union[int, slice], x: Union[ndarray,float]) -> None: self.__dataseries[i] = x
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
    def dataseries(self): return self.__dataseries
    @property
    def classname(self): return self.__class__.__name__
    @property
    def info(self):
        d={ 'class':self.classname,
            'rep':self.lengths,
            'dim':self.dim,
            'len':len(self.dataseries),
            'hom':self.homogeneous}
        return d
    @property
    def values(self): return self.mixture_to_one_dataseries_value()
    def mixture_to_one_dataseries_value(self): return mixture_to_one_dataseries_value(self)
    # -------------- MAGIC METHODS ----------------- #
    def __add__(self,other) -> Mixture: # for concatenation
        otherType = other.__class__.__name__
        if otherType == 'Mixture': return Mixture(self.dataseries + other.dataseries)
        elif otherType == 'DataSeries': return Mixture(self.dataseries + [other.dataseries])
        else: x = mixture(other) # will raise error if it does not qualify
        return self.__add__(self,x)
    def __radd__(left,self):
        leftType = left.__class__.__name__
        if leftType == 'DataSeries': return self.__add__(self,left)
        else: x = mixture(left)
        return self.__add__(self,x)



