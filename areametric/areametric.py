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

import numpy

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
    iqx, iqy = inverse_quantile_function(x,side='right'), inverse_quantile_function(y,side='right')
    if ((n1<n2) & (n2%n1!=0)) | ((n2<n1) & (n1%n2!=0)): # slow branch terminates in context
        xy = x + y # concatenate the two dataseries # sort happens here again
        xysv = xy.value_sorted # get data after sorting
        v = abs(iqx(xysv) - iqy(xysv)) # steps height
        u = diff(xysv) # steps width
        return sum(u*v[:-1]) # sum all area chunks and terminate
    elif (n2>n1) & (n2%n1==0): p = ecdf_p(y)
    elif (n1==n2) | ((n1>n2) & (n1%n2==0)): p = ecdf_p(x)
    p_= concatenate(([0.],p)) 
    pm = (p+p_[:-1])/2 # mid height of each step
    return sum(abs(qx(pm) - qy(pm)) / n)

    # if (n1==n2) | ((n1>n2) & (n1%n2==0)): p = ecdf_p(x)
    # elif (n2>n1) & (n2%n1==0): p = ecdf_p(y)
    # else: # datasets of different size 
        
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
    vv = abs(numpy.max(iqx_r,axis=0) - numpy.min(iqx_r,axis=0)) # steps height
    return numpy.sum((uu*vv[:-1]),axis=0) # sum all area chunks and terminate # transpose to allow broadcasting of multiplication

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
        

    # ii=[]
    # len_q = sum(x.repetitions)
    # shape_iq = tuple([len(x)]+[len_q]+list(x.dim))
    # iqxx_l,iqxx_r = empty(shape_iq),empty(shape_iq)
    # for i,xi in enumerate(x):
    #     if i==0: xx=xi  # todo: empty mixture 
    #     else: xx+=xi    # merge all data in one vector dataseries
    #     ii+=len(xi)*[i] # create corresponding vector of provenance indexes
    #     iqxx_l.append(inverse_quantile_function(xi,'left'))
    #     iqxx_r.append(inverse_quantile_function(xi,'right'))
    # xxsv = xx.value_sorted # get data after sorting
    # uu = abs(iqx(xxsv) - iqy(xxsv)) # steps height
    # vv = diff(xysv) # steps width
    # return sum(u[:-1]*v) # sum all area chunks and terminate

# def areame_mixture_tensor(x:ndarray,y:ndarray):
#     pass


# def areaMe_env(dataset: list) -> float:  # inputs a list of datasets of different sizes (list of lists)
#     '''
#     : --------------------------- ∞
#     cre: Oct 2020
#     edi: Mar 2022

#     web: github.com/marcodeangelis
#     org: University of Liverpool
    
#     MIT
#     : --------------------------- ∞

#     Code for the area metric of an envelope of datasets. Currenly under testing.
    
#     ''' 
#     def envelope(D: list): # inputs a list of datasets (list of lists)
#         F = [Dataset(d,bareclass=True).interp_f() for d in D]
#         dd =[]
#         ii =[]
#         for i,d in enumerate(D): 
#             dd+=d           # blend all data into a big dataset
#             ii+=len(d)*[i]  # assign a unique index to each element
#         dd.sort() # sort in place
#         I_min = []
#         I_max = []
#         d_prev = dd[0]
#         for d,i in zip(dd,ii):
#             Y_h = [f(d) for f in F]
#             v = (d_prev+d)/2
#             Y_l = [f(v) for f in F] # ensures interpolator on step picks the bottom value
#             ama = argmax(Y_h)
#             ami = argmin(Y_l)
#             I_max.append(ama)
#             I_min.append(ami)
#             d_prev = d
#         return I_min, I_max, F, dd
#     D = dataset.copy() # avoid catastrophic memory leaks due to the sort in place
#     minf,maxf,ff,dd = envelope(D) # -- main function starts here -- #
#     diff=[d1-d2 for d1,d2 in zip(dd[1:],dd[:-1])] # N-1 vector of differences
#     diff.sort()
#     for x in diff:
#         if x>0:
#             s=x/10 # this assigns to s the smallest difference greater than zero and makes it smaller
#             break
#     AM = 0
#     for i,_ in enumerate(dd):
#         if i<len(dd)-1:
#             d0 = dd[i] # = d
#             d1 = dd[i+1]
#             dx = d1-d0
#             dy = ff[maxf[i]](d0) - ff[minf[i+1]](d1-s)
#             AM += dx*dy
#     return AM      



# def areaMe(data1: DATASET_TYPE, data2: DATASET_TYPE) -> float: # inputs lists of doubles # numpy array are not currently supported.
#     '''
#     ∞ --------------------------- ∞
#     cre: Wed Oct 7 23:41 2020 
#     edi: Wed Jun 23 11:20 2021
#     author: Marco De Angelis 
#     github.com/marcodeangelis 
#     University of Liverpool 
#     GNU General Public License v3.0 
#     ∞ --------------------------- ∞
#     Works also on datasets of different sizes.
#     When datasets have same size the code is fastest.
#     '''
#     d1, d2 = dataset_parser(data1), dataset_parser(data2) 

#     n1, n2 = len(d1), len(d2)
#     f1_, f2_ = pseudoinverse(d1), pseudoinverse(d2)
#     if n1==n2:
#         n = n1 # = n2
#         p = numpy.linspace(0,1,n)
#         return sum(abs(f1_(p)-f2_(p))/n)
#     else: # n1>n2
#         # AM = 0
#         # f1, f2 = d1.interp_f(), d2.interp_f()
#         d12 = d1 + d2 
#         # s12 = d12.index()
#         v = d12.value()[d12.index()]
#         y = abs(f1_(v)-f2_(v))
#         x = numpy.diff(v)
#         return x[:-1]*y[:-1]

# def areaMe(data1: DATASET_TYPE, data2: DATASET_TYPE) -> float: # inputs lists of doubles # numpy array are not currently supported.
#     '''
#     ∞ --------------------------- ∞
#     cre: Wed Oct 7 23:41 2020 
#     edi: Wed Jun 23 11:20 2021
#     author: Marco De Angelis 
#     github.com/marcodeangelis 
#     University of Liverpool 
#     GNU General Public License v3.0 
#     ∞ --------------------------- ∞

#     Works also on datasets of different sizes.

#     When datasets have same size the code is fastest.

#     '''
#     d1, d2 = dataset_parser(data1), dataset_parser(data2) 

#     n1, n2 = len(d1), len(d2)
#     f1_, f2_ = pseudoinverse(d1), pseudoinverse(d2)
#     if n1==n2:
#         n = n1 # = n2
#         p = numpy.linspace(0,1,n)
#         return sum(abs(f1_(p)-f2_(p))/n)
#     else: # n1>n2
#         # AM = 0
#         # f1, f2 = d1.interp_f(), d2.interp_f()
#         d12 = d1 + d2 
#         # s12 = d12.index()
#         v = d12.value()[d12.index()]
#         y = abs(f1_(v)-f2_(v))
#         x = numpy.diff(v)
#         return x[:-1]*y[:-1]
#     #     for i in range(n1+n2-1):
#     #         v = d12[s12[i]] # v = (d12[s12[i]]+d12[s12[i+1]])/2
#     #         y = abs(f1(v)-f2(v))
#     #         x = d12[s12[i+1]]-d12[s12[i]]
#     #         AM += x*y
#     # return AM


# NUMERIC_TYPES = ('int','float','complex','int8','int16','int32','int64','float16','float32','float64','complex128') # not comprehensive

# ARRAY_TYPE = type(numpy.empty(0))

# def show(x: Iterable, N: int=5) -> str:
#     if len(x)>10:
#         a1,a2 = [str(i) for i in x[:N]], [str(i) for i in x[-N:]]
#         return '\n'.join(a1+['...']+a2)
#     else:
#         return '\n'.join([str(i) for i in x])


# class Dataset(): # created for code safety
#     '''
#     Mon Jan 4 18:33 2020 
#     @author: Marco De Angelis 
#     '''
#     def __repr__(self): # return
#         return show(self)
#     def __str__(self): # print
#         return show(self)
#     def __iter__(self): # make class iterable
#         for v in self.__data:
#             yield v
#     def __getitem__(self,index): # make class subscrictable
#         if index.__class__.__name__ in ['list','tuple']:
#             if len(index)>1:
#                 return Dataset([self.__data[i] for i in index])
#             elif len(index)==1:
#                 return self.__data[index[0]]
#             elif len(index)==0:
#                 return None
#         else:
#             return self.__data[index]
#     def __len__(self):
#         return len([i for i in self])
#     def __init__(self,data: list,name='x',bareclass=False):
#         if not bareclass: # set this to True if needed for speed
#             self.name = name
#             if self.is_iterable(data):    # assert type(data)==list, 'The dataset is constructed from a python list of numbers. \nSupport for numpy.array is not yet available.'
#                 for d in data: # this check can be lenghty O(N)
#                     self.is_numeric(d) # throws an error if the iterable does not contain number types
#                 self.__data, self.__ordered = list(data.copy()), list(data.copy()) # make two copies of the dataset
#                 self.__ordered.sort() 
#             else:
#                 self.is_numeric(data) # throws an error if it is not a number
#                 self.__data, self.__ordered = list([data]), list([data])
#     def __add__(self,other):
#         othertype = other.__class__.__name__
#         if othertype == 'list':
#             return Dataset(self.__data + other)
#         elif othertype == 'tuple':
#             return Dataset(self.__data + list(other))
#         elif isinstance(other,self.__class__):
#             return Dataset(self.__data + other.to_list())
#     def __sub__(self,other):
#         if isinstance(other,self.__class__):
#             return areaMe(self.__data,other.to_list())
#         else:
#             return TypeError('Allowed only between two Dataset types.')

#     def __eq__(self,other):
#         for s,o in zip(self,other):
#             if s!=o:
#                 return False
#         return True

#     def to_list(self):
#         return self.__data
#     def to_array(self):
#         return numpy.array(self.__data)
    
#     @staticmethod
#     def is_iterable(data) -> bool:
#         try:
#             for v in data:
#                 return True # if iterable exit at first iteration O(1), else except O(1).
#         except TypeError as e:
#             number_type = data.__class__.__name__
#             expected_error = f"'{number_type}' object is not iterable"
#             if str(e) == expected_error:
#                 print(f'{e}. Convertion to the single element list: [{data}]')
#             else:
#                 print('something went wrong.') # this should never be reached.
#             return False

#     @staticmethod
#     def is_numeric(x):
#         if x.__class__.__name__ not in NUMERIC_TYPES:
#             print(f"Numeric types currently supported {NUMERIC_TYPES}")
#             raise TypeError('The elements in the list must be numeric.')
#         return None
    
#     def ecdf(self):
#         x = self.__ordered
#         n = len(x)
#         p = 1/n
#         pvalues = list(linspace(p,1,n))
#         return x, pvalues
#     def stairs(self):
#         def stepdata(x,y): # x,y must be python lists
#             xx,yy = x*2, y*2
#             xx.sort()
#             yy.sort()
#             return xx, [0.]+yy[:-1]
#         x, p = self.ecdf()
#         x, y = stepdata(x,p)
#         return x, y
#     def interp_f(self):
#         from scipy.interpolate import interp1d   # this is needed only when n1!=n2
#         n = len(self)
#         if n==1:
#             n,d = 2,2*self.__data
#         else:
#             d = self.__ordered
#         p = linspace(1/n,1,n)
#         f = interp1d(d, p, kind='previous')
#         def wrapper(x):
#             try:
#                 return f(x)
#             except ValueError as e: 
#                 if str(e)=='A value in x_new is above the interpolation range.':
#                     return 1    
#                 elif str(e) == 'A value in x_new is below the interpolation range.':
#                     return 0
#         return wrapper
#     # def argsort(self):
#     #     return numpy.argsort(self.__data) # outputs a numpy array


# class ParametricDataset(Dataset): 
#     def __init__(self,distribution,N=100):
#         y=numpy.linspace(0.001,0.999,N)        
#         super().__init__(list(distribution.ppf(y)))
#     def superclass(self):
#         return self.__class__.__bases__[0].__name__


# def areaMe_(data1: list, data2: list) -> float: # inputs lists of doubles # numpy array are not currently supported.
#     '''
#     ∞ --------------------------- ∞
#     Wed Oct 7 23:41:25 2020 
#     @author: Marco De Angelis 
#     github.com/marcodeangelis 
#     University of Liverpool 
#     GNU General Public License v3.0 
#     ∞ --------------------------- ∞

#     Works also on datasets of different sizes.

#     When datasets have same size the code is fastest.

#     '''
#     # d1,d2 = data1.copy(), data2.copy() # avoid catastrophic memory leaks due to sort in place
#     # assert type(data1)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
#     # assert type(data2)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
#     d1, d2 = Dataset(data1), Dataset(data2) # also checks dataset is numeric and iterable

#     n1 = len(d1)
#     n2 = len(d2)
#     if n1==n2:
#         x1,_ = d1.stairs()
#         x2,_ = d2.stairs()
#         AM = sum([abs(l-r) for r,l in zip(x1,x2)])/(2*n1)
#     else: # n1>n2
#         AM = 0
#         f1, f2 = d1.interp_f(), d2.interp_f()
#         d12 = d1 + d2  # Need to be lists!
#         s12 = d12.argsort()
#         for i in range(n1+n2-1):
#             v = d12[s12[i]] # v = (d12[s12[i]]+d12[s12[i+1]])/2
#             y = abs(f1(v)-f2(v))
#             x = d12[s12[i+1]]-d12[s12[i]]
#             AM += x*y
#     return AM