'''
    %---------------------------------------#
    Thu Oct 15 22:15:39 2020
    @author: Marco De Angelis & Jaleena Sunny
    github.com/marcodeangelis
    University of Liverpool
    GNU General Public License v3.0
    %---------------------------------------#

    Fast code for area metric. This version works with data sets of different sizes.

    When datasets have same size the code is fastest.

    Updated version of the AM for an envelope of datasets.

    % --- About GNU General Public License v3.0 --- #
    Permissions of this strong copyleft license are conditioned on making available complete source code 
    of licensed works and modifications, which include larger works using a licensed work, under the same license. 
    Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
    '''

import numpy
from numpy import argsort, linspace, argmax, argmin
from scipy.interpolate import interp1d   # this is needed only when n1!=n2
from matplotlib import pyplot


NUMERIC_TYPES = ('int','float','complex','int8','int16','int32','int64','float16','float32','float64','complex128') # not comprehensive

class Dataset(): # created for code safety
    '''
    ∞ --------------------------- ∞
    Mon Jan 4 18:33 2020 
    @author: Marco De Angelis 
    github.com/marcodeangelis 
    University of Liverpool 
    GNU General Public License v3.0 
    ∞ --------------------------- ∞
    '''
    def __repr__(self): # return
        if len(self)>10:
            a = [str(i) for i in self]
            s = '\n'.join(a[:5]+['...']+a[-5:-1]) # display only the first and last 5 items
        else:
            s = '\n'.join([str(i) for i in self])
        return s
    def __str__(self): # print
        s = '\n'.join([str(i) for i in self])
        return s
    def __iter__(self): # make class iterable
        for v in self.__data:
            yield v
    def __getitem__(self,index): # make class subscrictable
        if index.__class__.__name__ in ['list','tuple']:
            if len(index)>1:
                return Dataset([self.__data[i] for i in index])
            elif len(index)==1:
                return self.__data[index[0]]
            elif len(index)==0:
                return None
        else:
            return self.__data[index]
    def __len__(self):
        return len([i for i in self])
    def __init__(self,data: list,name='x',bareclass=False):
        if not bareclass: # set this to True if needed for speed
            self.name = name
            assert type(data)==list, 'The dataset is constructed from a python list of numbers. \nSupport for numpy.array is not yet available.'
            self.is_numeric(data)
        self.__data, self.__ordered = list(data.copy()), list(data.copy()) # make two copies of the dataset
        self.__ordered.sort() 
    def __add__(self,other):
        othertype = other.__class__.__name__
        if othertype == 'list':
            return Dataset(self.__data + other)
        elif othertype == 'tuple':
            return Dataset(self.__data + list(other))
        elif isinstance(other,self.__class__):
            return Dataset(self.__data + other.to_list())
    def __sub__(self,other):
        if isinstance(other,self.__class__):
            return areaMe(self.__data,other.to_list())
        else:
            return TypeError('Allowed only between two Dataset types.')

    def to_list(self):
        return self.__data
    def to_array(self):
        return numpy.array(self.__data)
    
    @staticmethod
    def is_numeric(data):
        for v in data:
            if v.__class__.__name__ not in NUMERIC_TYPES:
                print(f"Numeric types currently supported {NUMERIC_TYPES}")
                raise TypeError('The elements in the list must be numeric.')
        return None
    
    def ecdf(self) -> (list, list):
        x = self.__ordered
        n = len(x)
        p = 1/n
        pvalues = list(linspace(p,1,n))
        return x, pvalues
    def stairs(self) -> (list, list):
        def stepdata(x,y): # x,y must be python lists
            xx,yy = x*2, y*2
            xx.sort()
            yy.sort()
            return xx, [0.]+yy[:-1]
        x, p = self.ecdf()
        x, y = stepdata(x,p)
        return x, y
    def interp_f(self):
        n = len(self)
        if n==1:
            n,d = 2,2*self.__data
        else:
            d = self.__ordered
        p = linspace(1/n,1,n)
        f = interp1d(d, p, kind='previous')
        def wrapper(x):
            try:
                return f(x)
            except ValueError as e: 
                if str(e)=='A value in x_new is above the interpolation range.':
                    return 1    
                elif str(e) == 'A value in x_new is below the interpolation range.':
                    return 0
        return wrapper
    def argsort(self):
        return numpy.argsort(self.__data) # outputs a numpy array
    def plot(self,ax=None, figsize=(12,8), marker='.', lw=1, fontsize='16', xlabel=None, ylabel='Probability', title='Empirical cumulative distribution'):
        if ax is None:
            _, ax = pyplot.subplots(figsize=figsize)
            ax.grid()
        x,y = self.stairs()
        ax.plot(x,y,lw=lw,marker=marker)
        ax.set_title(title,fontsize=str(int(fontsize)+4))
        if xlabel is None:
            ax.set_xlabel(self.name,fontsize=fontsize)
        else:
            ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        pass

class ParametricDataset(Dataset): 
    def __init__(self,distribution,N=100):
        y=numpy.linspace(0.001,0.999,N)        
        super().__init__(list(distribution.ppf(y)))
    def superclass(self):
        return self.__class__.__bases__[0].__name__

def plot(data1, data2, ax=None, alpha=0.3, color='gray', figsize=(12,8), marker='.', lw=1, fontsize='16', xlabel='d1, d2', ylabel='Probability', title=None,legend=True, areait=True, savefig=None):
    assert type(data1)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
    assert type(data2)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
    d1, d2 = Dataset(data1), Dataset(data2)

    n1,n2 = len(d1),len(d2)

    x1,y1 = d1.stairs()
    x2,y2 = d2.stairs()

    if ax is None:
        fig, ax = pyplot.subplots(figsize=figsize)
        ax.grid()

    ax.plot(x1,y1,lw=lw,marker=marker,label='d1')
    ax.plot(x2,y2,lw=lw,marker=marker,label='d2')
    if n1 == n2:
        ax.fill_betweenx(y1,x1,x2, alpha=alpha, color=color)

    ax.set_title(title,fontsize=str(int(fontsize)+4))
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    if legend:
        ax.legend()
    if areait:
        am = areaMe(data1,data2)
        xt = (x1[1]+x1[2])/2
        yt = (y1[0]+y1[1])/2
        ax.text(xt,yt,f"area = {'%g'%am}")
    if savefig is not None:
        fig.savefig(savefig)
    pass

def subplots(figsize=(12,8)): # wrapper to avoid importing matplotlib in applications
    return pyplot.subplots() # returns (fig, ax) 

def areaMe(data1: list, data2: list) -> float: # inputs lists of doubles # numpy array are not currently supported.
    '''
    ∞ --------------------------- ∞
    Wed Oct 7 23:41:25 2020 
    @author: Marco De Angelis 
    github.com/marcodeangelis 
    University of Liverpool 
    GNU General Public License v3.0 
    ∞ --------------------------- ∞

    Fast code for area metric. This version works with data sets of different sizes.

    When datasets have same size the code is fastest.

    '''
    # d1,d2 = data1.copy(), data2.copy() # avoid catastrophic memory leaks
    assert type(data1)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
    assert type(data2)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
    d1, d2 = Dataset(data1), Dataset(data2) # also checks dataset is numeric

    n1 = len(d1)
    n2 = len(d2)
    if n1==n2:
        x1,_ = d1.stairs()
        x2,_ = d2.stairs()
        AM = sum([abs(l-r) for r,l in zip(x1,x2)])/(2*n1)
    else: # n1>n2
        AM = 0
        f1, f2 = d1.interp_f(), d2.interp_f()
        d12 = d1 + d2  # Need to be lists!
        s12 = d12.argsort()
        for i in range(n1+n2-1):
            v = d12[s12[i]] # v = (d12[s12[i]]+d12[s12[i+1]])/2
            y = abs(f1(v)-f2(v))
            x = d12[s12[i+1]]-d12[s12[i]]
            AM += x*y
    return AM

def areaMe_env(dataset: list) -> float:  # inputs a list of datasets of different sizes (list of lists)
    '''
    ∞ --------------------------- ∞
    Fri Oct 9 18:53:08 2020
    @author: Marco De Angelis
    github.com/marcodeangelis
    University of Liverpool
    GNU General Public License v3.0
    ∞ --------------------------- ∞

    Code for the area metric of an envelope of datasets. Currenly under testing.
    
    ''' 
    def envelope(D: list) -> (list, list, list, list): # inputs a list of datasets (list of lists)
        F = [Dataset(d,bareclass=True).interp_f() for d in D]
        dd =[]
        ii =[]
        for i,d in enumerate(D): 
            dd+=d           # blend all data into a big dataset
            ii+=len(d)*[i]  # assign a unique index to each element
        dd.sort() # sort in place
        I_min = []
        I_max = []
        d_prev = dd[0]
        for d,i in zip(dd,ii):
            Y_h = [f(d) for f in F]
            v = (d_prev+d)/2
            Y_l = [f(v) for f in F] # ensures interpolator on step picks the bottom value
            ama = argmax(Y_h)
            ami = argmin(Y_l)
            I_max.append(ama)
            I_min.append(ami)
            d_prev = d
        return I_min, I_max, F, dd
    D = dataset.copy() # avoid catastrophic memory leaks due to the sort in place
    minf,maxf,ff,dd = envelope(D) # -- main function starts here -- #
    diff=[d1-d2 for d1,d2 in zip(dd[1:],dd[:-1])] # N-1 vector of differences
    diff.sort()
    for x in diff:
        if x>0:
            s=x/10 # this assigns to s the smallest difference greater than zero and makes it smaller
            break
    AM = 0
    for i,_ in enumerate(dd):
        if i<len(dd)-1:
            d0 = dd[i] # = d
            d1 = dd[i+1]
            dx = d1-d0
            dy = ff[maxf[i]](d0) - ff[minf[i+1]](d1-s)
            AM += dx*dy
    return AM      