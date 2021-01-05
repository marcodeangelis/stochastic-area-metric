'''
    %---------------------------------------#
    Thu Nov 18 19:57 2020
    @author: Marco De Angelis & Jaleena Sunny
    github.com/marcodeangelis
    University of Liverpool
    GNU General Public License v3.0
    %---------------------------------------#

    A wrapper of the scipy stats library for probability distributions with additional plotting.

'''

import scipy.stats as scipystats
import numpy   
from matplotlib import pyplot


def ecdf(x: list): # x must be python list
    n = len(x)
    p = 1/n
    x.sort() # careful.. sort in place
    pvalues = list(numpy.linspace(p,1,n))
    return x, pvalues

def plotdist(dist, N=30, ax=None, figsize=(16,9), savefig=None, fontsize=16):
    if ax is None:
        fig, ax = pyplot.subplots(2,2,figsize=figsize,sharex=True)
        x = numpy.linspace(dist.ppf(0.001),dist.ppf(0.999), 100)
    y1 = dist.pdf(x)
    y2 = dist.cdf(x)
    ax[0,0].plot(x,y1,lw=3,c="#424242")
    ax[0,1].plot(x,y1,lw=3,c="#424242")
    ax[1,0].plot(x,y2,lw=3,c="#424242")
    ax[1,1].plot(x,y2,lw=1,c="#424242")
    line0 = [[dist.ppf(0.001),dist.ppf(0.001)],[0,dist.pdf(dist.ppf(0.001))]]
    line1 = [[dist.ppf(0.999),dist.ppf(0.999)],[0,dist.pdf(dist.ppf(0.999))]]
    ax[0,0].plot(line0[0],line0[1],":k")
    ax[0,0].plot(line1[0],line1[1],":k")
    r = dist.rvs(N)
    ax[0,1].hist(r, density=True, histtype='stepfilled', alpha=0.5)
    xs, ys = ecdf(list(r)) # This require the ecdf in this file
    ax[1,1].scatter(xs,ys,marker=".")
    ax[0,0].set_ylabel("PDF",fontsize=fontsize,color="#5a5a64")
    ax[1,0].set_ylabel("CDF",fontsize=fontsize,color="#5a5a64")
    ax[0,1].set_ylabel("Histogram & PDF",fontsize=fontsize,color="#5a5a64")
    ax[1,1].set_ylabel("ECDF",fontsize=fontsize,color="#5a5a64")
    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlabel(dist.name,fontsize=fontsize,color="#5a5a64")
            ax[i,j].grid(b=True)
            ax[i,j].tick_params(direction='out', length=6, width=2, colors='#5a5a64',grid_color='b', grid_alpha=0.5, labelsize='x-large')
    s1 = "%s ~ %s(t1=%g, t2=%g)"%(dist.name,dist.disttype(),dist.theta_1(),dist.theta_2())
    s2 = "%s ~ %s(m=%g, s=%g)"%(dist.name,dist.disttype(),dist.mean(),dist.std())
    ax[0,0].set_title(s1,fontsize=fontsize,color="#5a5a64")
    ax[0,1].set_title(s2,fontsize=fontsize,color="#5a5a64")
    pyplot.tight_layout()
    if savefig is not None:
        fig.savefig(savefig)
    pass
    

class Distribution():
    def __repr__(self): # return
        return "%s ~ %s(t1=%g, t2=%g)"%(self.name,self.__disttype,self.__theta1,self.__theta2)

    def __str__(self): # print
        return "%s ~ %s(m=%g, s=%g)"%(self.name,self.__disttype,self.mean(),self.std())

    def __init__(self,**kwargs):
        self.name = "" # initialise this with an empty string
        for key, value in kwargs.items():
            if key == "name":
                self.name = value
            elif key == "disttype":
                self.__disttype = value
            elif key == "thetas":
                self.__theta1 = value[0]
                self.__theta2 = value[1]
            elif key == "moments":
                self.__moments = value
            elif key == "scipy":
                self.__scipystats = value
            elif key == "locscale":
                self.__loc = value[0]
                self.__scale = value[1]
    def mean(self):
        return self.__scipystats.mean(loc=self.__loc,scale=self.__scale)
    def variance(self):
        return self.__scipystats.var(loc=self.__loc,scale=self.__scale)
    def std(self):
        return self.__scipystats.std(loc=self.__loc,scale=self.__scale)
    def moment(self,n=1):
        return self.__scipystats.moment(n,loc=self.__loc,scale=self.__scale)
    def stats(self,m='mv'):
        return self.__scipystats.stats(loc=self.__loc, scale=self.__scale, moments=m)
    def median(self):
        return self.__scipystats.median(loc=self.__loc,scale=self.__scale)
    def interval(self,a=0.5):
        return self.__scipystats.interval(a,loc=self.__loc,scale=self.__scale)
    def family(self):
        return self.__disttype
    def disttype(self):
        return self.__disttype
    def theta_1(self):
        return self.__theta1
    def theta_2(self):
        return self.__theta2
    def cdf(self,x):
        return self.__scipystats.cdf(x,loc=self.__loc,scale=self.__scale)
    def pdf(self,x):
        return self.__scipystats.pdf(x,loc=self.__loc,scale=self.__scale)
    def ppf(self,q):
        return self.__scipystats.ppf(q,loc=self.__loc,scale=self.__scale)
    def sample(self,N=1):
        from numpy import random as rnd
        r = rnd.random_sample(N)
        return self.__scipystats.ppf(r,loc=self.__loc,scale=self.__scale)
    def rvs(self,n=1):
        return self.__scipystats.rvs(loc=self.__loc,scale=self.__scale,size=n)
    def plot(self,N=30,ax=None,figsize=(16,9),savefig=None,fontsize=16):
        plotdist(self,N=N,ax=None,figsize=figsize,savefig=None,fontsize=fontsize)
        pass
    
    
class Distribution1(): # This are distributions that require only one parameter to be identified, like the exponential distribution
    def __repr__(self): # return
        return "%s ~ %s(t1=%g)"%(self.name,self.__disttype,self.__theta1)

    def __str__(self): # print
        return "%s ~ %s(m=%g)"%(self.name,self.__disttype,self.mean())

    def __init__(self,**kwargs):
        self.name = "" # initialise this with an empty string
        for key, value in kwargs.items():
            if key == "name":
                self.name = value
            elif key == "disttype":
                self.__disttype = value
            elif key == "thetas":
                self.__theta1 = value[0]
            elif key == "additional":
                self.__s = value
            elif key == "scipy":
                self.__scipystats = value
            elif key == "locscale":
                self.__loc = value[0]
                self.__scale = value[1]
    def mean(self):
        return self.__scipystats.mean(loc=self.__loc,scale=self.__scale)
    def variance(self):
        return self.__scipystats.var(loc=self.__loc,scale=self.__scale)
    def std(self):
        return self.__scipystats.std(loc=self.__loc,scale=self.__scale)
    def moment(self,n=1):
        return self.__scipystats.moment(n,loc=self.__loc,scale=self.__scale)
    def stats(self,m='mv'):
        return self.__scipystats.stats(loc=self.__loc, scale=self.__scale, moments=m)
    def median(self):
        return self.__scipystats.median(loc=self.__loc,scale=self.__scale)
    def interval(self,a=0.5):
        return self.__scipystats.interval(a,loc=self.__loc,scale=self.__scale)
    def family(self):
        return self.__disttype
    def disttype(self):
        return self.__disttype
    def theta_1(self):
        return self.__theta1
    def loc(self):
        return self.__loc
    def scale(self):
        return self.__scale
    def shape(self):
        return self.__s
    def cdf(self,x):
        return self.__scipystats.cdf(x,loc=self.__loc,scale=self.__scale)
    def pdf(self,x):
        return self.__scipystats.pdf(x,loc=self.__loc,scale=self.__scale)
    def ppf(self,q):
        return self.__scipystats.ppf(q,loc=self.__loc,scale=self.__scale)
    def sample(self,N=1):
        r = numpy.random.random_sample(N)
        return self.__scipystats.ppf(r,loc=self.__loc,scale=self.__scale)
    def rvs(self,n=1):
        return self.__scipystats.rvs(loc=self.__loc,scale=self.__scale,size=n)
    def plot(self,N=30,ax=None,figsize=(16,9),savefig=None,fontsize=16):
        plotdist(self,N=N,ax=None,figsize=figsize,savefig=None,fontsize=fontsize)
        pass
    
    
class Distribution2(): # This are distributions that require an additional parameter beyond loc and scale 
    def __repr__(self): # return
        return "%s ~ %s(t1=%g, t2=%g)"%(self.name,self.__disttype,self.__theta1,self.__theta2)

    def __str__(self): # print
        return "%s ~ %s(m=%g, s=%g)"%(self.name,self.__disttype,self.mean(),self.std())

    def __init__(self,**kwargs):
        self.name = "" # initialise this with an empty string
        for key, value in kwargs.items():
            if key == "name":
                self.name = value
            elif key == "disttype":
                self.__disttype = value
            elif key == "thetas":
                self.__theta1 = value[0]
                self.__theta2 = value[1]
            elif key == "additional":
                self.__s = value
            elif key == "scipy":
                self.__scipystats = value
            elif key == "locscale":
                self.__loc = value[0]
                self.__scale = value[1]
    def mean(self):
        return self.__scipystats.mean(self.__s,loc=self.__loc,scale=self.__scale)
    def variance(self):
        return self.__scipystats.var(self.__s,loc=self.__loc,scale=self.__scale)
    def std(self):
        return self.__scipystats.std(self.__s,loc=self.__loc,scale=self.__scale)
    def moment(self,n=1):
        return self.__scipystats.moment(n,self.__s,loc=self.__loc,scale=self.__scale)
    def stats(self,m='mv'):
        return self.__scipystats.stats(self.__s,loc=self.__loc, scale=self.__scale, moments=m)
    def median(self):
        return self.__scipystats.median(self.__s,loc=self.__loc,scale=self.__scale)
    def interval(self,a=0.5):
        return self.__scipystats.interval(a,self.__s,loc=self.__loc,scale=self.__scale)
    def family(self):
        return self.__disttype
    def disttype(self):
        return self.__disttype
    def theta_1(self):
        return self.__theta1
    def theta_2(self):
        return self.__theta2
    def loc(self):
        return self.__loc
    def scale(self):
        return self.__scale
    def shape(self):
        return self.__s
    def cdf(self,x):
        return self.__scipystats.cdf(x,self.__s,loc=self.__loc,scale=self.__scale)
    def pdf(self,x):
        return self.__scipystats.pdf(x,self.__s,loc=self.__loc,scale=self.__scale)
    def ppf(self,q):
        return self.__scipystats.ppf(q,self.__s,loc=self.__loc,scale=self.__scale)
    def sample(self,N=1):
        r = numpy.random.random_sample(N)
        return self.__scipystats.ppf(r,self.__s,loc=self.__loc,scale=self.__scale)
    def rvs(self,n=1):
        return self.__scipystats.rvs(self.__s,loc=self.__loc,scale=self.__scale,size=n) 
    def plot(self,N=30,ax=None,figsize=(16,9),savefig=None,fontsize=16):
        plotdist(self,N=N,ax=None,figsize=figsize,savefig=None,fontsize=fontsize)
        pass
    

class uniform(Distribution):
    def __init__(self,*args, name='x'):
        self.__scipystats = scipystats.uniform
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__theta1 = arg
                elif i == 1:
                    self.__theta2 = arg
                i += 1
#         self.__m1 = 0.5 * (self.__theta1 + self.__theta2)
#         self.__m2 = (1/12) * (self.__theta2 - self.__theta1)**2
        self.__loc = self.__theta1
        self.__scale = self.__theta2 - self.__theta1
        super().__init__(disttype='uniform',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    
    
class normal(Distribution):
    def __init__(self,*args,name='x'):
        self.__scipystats = scipystats.norm
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__theta1 = arg
                elif i == 1:
                    self.__theta2 = arg
                i += 1
        self.__loc = self.__theta1 # The mean and location coincides for the normal dist
        self.__scale = self.__theta2  # The standard deviation and the scale coincides for the normal dist
        super().__init__(disttype='normal',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    
    
class lognormal(Distribution2): # inputs the parameters
    def __init__(self,*args,name='x') -> "inputs the parameters":
        self.__scipystats = scipystats.lognorm
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__theta1 = arg
                elif i == 1:
                    self.__theta2 = arg
                i += 1
        self.__loc = 0   # location  # The location of the lognormal controls the shift of the distribution.
        self.__s = self.__theta2 # shape   # This is the standard deviation of the normally distributed random variable X such that Y=log(X)
        self.__scale = numpy.exp(self.__theta1)   # scale # This is the exponential of the mean of Y, exp(mu), where mu=mean(x).
        
        super().__init__(disttype='lognormal',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale], additional=self.__s,\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    
    
class Lognormal(Distribution2): 
    def __init__(self,*args,name='x') -> "inputs the moments (mean and std)":
        self.__scipystats = scipystats.lognorm
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__mean_of_the_lognormal = arg
                elif i == 1:
                    self.__std_of_the_lognormal = arg
                i += 1   
        self.__theta1, self.__theta2 = self.ms_to_t1t2()
        self.__loc = 0   # location  # The location of the lognormal controls the shift of the distribution.
        self.__s = self.__theta2 # shape   # This is the standard deviation of the normally distributed random variable X such that Y=log(X)
        self.__scale = numpy.exp(self.__theta1)   # scale # This is the exponential of the mean of Y, exp(mu), where mu=mean(x).
        
        super().__init__(disttype='Lognormal',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale], additional=self.__s,\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    def ms_to_t1t2(self):
        m = self.__mean_of_the_lognormal
        s = self.__std_of_the_lognormal
        r2 = (s/m)**2
        t1 = numpy.log(m/numpy.sqrt(1+r2))
        t2 = numpy.sqrt(numpy.log(1+r2))
        return t1,t2
    
    
class gumbel(Distribution): # inputs the parameters
    def __init__(self,*args,name='x'):
        self.__scipystats = scipystats.gumbel_r
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__theta1 = arg
                elif i == 1:
                    self.__theta2 = arg
                i += 1
        self.__loc = self.__theta1 
        self.__scale = self.__theta2  
        super().__init__(disttype='gumbel_max',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    
    
class Gumbel(Distribution):
    def __init__(self,*args,name='x'): # inputs the moments (mean and std)
        self.__scipystats = scipystats.gumbel_r
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__mean_of_the_gumbel = arg
                elif i == 1:
                    self.__std_of_the_gumbel = arg
                i += 1
        self.__theta1, self.__theta2 = self.ms_to_t1t2()
        self.__loc = self.__theta1 
        self.__scale = self.__theta2  
        super().__init__(disttype='gumbel_max',thetas=[self.__theta1,self.__theta2],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    def ms_to_t1t2(self):
        m = self.__mean_of_the_gumbel
        s = self.__std_of_the_gumbel
        t1 = m - numpy.euler_gamma * numpy.sqrt(6) * s / numpy.pi
        t2 = numpy.sqrt(6) * s / numpy.pi
        return t1,t2
    
    
class exponential(Distribution1):
    def __init__(self,*args,name='x'):
        self.__scipystats = scipystats.expon
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__theta1 = arg
                i += 1
        self.__loc = 0 
        self.__scale = 1/self.__theta1
        super().__init__(disttype='exponential',thetas=[self.__theta1],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    
    
class Exponential(Distribution1):
    def __init__(self,*args,name='x'):
        self.__scipystats = scipystats.expon
        self.name = name 
        i = 0
        for arg in args:
            if arg.__class__.__name__ == "str":
                self.name = arg
            else:
                if i == 0:
                    self.__mean_of_the_exponential = arg[i]
                i += 1
        self.__theta1 = self.m_to_t1()
        self.__loc = 0 
        self.__scale = 1/self.__theta1
        super().__init__(disttype='Exponential',thetas=[self.__theta1],\
                         locscale=[self.__loc,self.__scale],\
                         scipy = self.__scipystats,name=self.name)
    def superclass(self):
        return self.__class__.__bases__[0].__name__
    def m_to_t1(self):
        m = self.__mean_of_the_exponential
        t1 = 1/m
        return t1