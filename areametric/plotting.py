'''
-------------------------------
Created Feb 2022
github.com/marcodeangelis
University of Liverpool
GNU General Public License v3.0
-------------------------------
'''
from __future__ import annotations
from tkinter import font
from typing import Sequence, Sized, Union, Iterable, Optional, Any, Callable, Tuple

import numpy
from numpy import (asarray, empty, ndarray, sort, concatenate, diff, zeros, ones)

from matplotlib import pyplot

from .dataseries import dataseries
from .areametric import areaMe
from .methods import (ecdf,ecdf_, ecdf_p, quantile_function, inverse_quantile_function, inverse_quantile_value)

FONTSIZE = 16
FIGSIZE = (12,8)

class ECDF_BOXED():
    count=0
    def __init__(self): None
    def increment(self): ECDF_BOXED.count+=1
    def reset(self): ECDF_BOXED.count=0

class AREA_PLOTS():
    count=0
    def __init__(self): None
    def increment(self): AREA_PLOTS.count+=1
    def reset(self): AREA_PLOTS.count=0

C_ECDF_BOXED = ECDF_BOXED()
C_AREA_PLOTS = AREA_PLOTS()


# https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html
# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

COLOR_SEQUENCE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # Python default color sequence

def stairs(x:ndarray, above:bool=True):
    def stepdata(x,y): 
        xx,yy = sort(concatenate((x,x))), sort(concatenate((y,y))) # sort(2*x), sort(2*y)
        if above: return xx, concatenate(([0.],yy[:-1]))
        else: return concatenate(([xx[0]],xx[:-1])), yy
    x,p = ecdf(x) # x is parsed in ecdf
    x,y = stepdata(x,p)
    return x, y

def plot_ecdf(x_:ndarray,ax=None,figsize=FIGSIZE,fontsize:int=FONTSIZE,grid=False,color=None):
    x = dataseries(x_).value
    fig=[]
    if ax is None:fig,ax = pyplot.subplots(figsize=figsize)
    if grid: ax.grid()
    x1,y1 = stairs(x)
    ax.plot(x1,y1,color=color)
    ax.scatter(x,[0]*len(x),marker='|',s=200) # https://matplotlib.org/stable/api/markers_api.html
    x2,p=ecdf(x)
    ax.scatter(x2,p,s=10,color='red',zorder=3)
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('p',fontsize=fontsize)
    ax.tick_params(direction='out', grid_alpha=0.5, labelsize='large')
    return fig, ax 

def plot_ecdf_boxed(x_:ndarray,ax=None,figsize=FIGSIZE,fontsize:int=FONTSIZE,grid=False,midline:bool=False):
    x = dataseries(x_).value
    fig=[]
    if ax is None: 
        fig,ax = pyplot.subplots(figsize=figsize)
        if grid: ax.grid()
        C_ECDF_BOXED.reset() 
    else: C_ECDF_BOXED.increment() # increment counter to move on to the next color in the sequence. 
    color = COLOR_SEQUENCE[C_ECDF_BOXED.count%10]
    x1,y1 = stairs(x,above=True)
    x2,y2 = stairs(x,above=False)
    ax.plot(x1,y1,color=color)
    ax.plot(x2,y2,color=color)
    ax.fill_between(x=x1,y1=y1,y2=y2,color=color,alpha=0.05)
    ax.fill_between(x=x2,y1=y1,y2=y2,color=color,alpha=0.05)
    if midline:
        x3,y3=ecdf_(x)
        ax.plot(x3,y3, color=color,lw=0.2)
    ax.scatter(x,[0]*len(x),marker='|',s=200)
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('p',fontsize=fontsize)
    ax.tick_params(direction='out', grid_alpha=0.5, labelsize='large') #,colors='#5a5a64'
    return fig, ax 

def plot_box(a:ndarray,b:ndarray,ax:pyplot=None,figsize=FIGSIZE,grid:bool=False,
                                facecolor=None,edgecolor=None,alpha=0.15,label=None,zorder=None): # plot the box from intervals a and b
    '''
    INPUTS

    a: 1x2 ndarray 
        x-interval
    
    b: 1x2 ndarray
        y-interval
    '''
    fig=None
    if ax is None: 
        fig,ax = pyplot.subplots(figsize=figsize)
        if grid: ax.grid()
    ax.fill_between(x=[a[0],a[1]], y1=[b[0],b[0]], y2=[b[1],b[1]],facecolor=facecolor,edgecolor=edgecolor,alpha=alpha,label=label,zorder=zorder)
    return fig,ax


def plot_area(x_:ndarray,y_:ndarray,ax=None,alpha:float=0.15,color:str='gray', figsize:tuple=FIGSIZE, marker:str='.', 
                                   lw:float=1,fontsize:str=FONTSIZE,xlabel:str='x, y',ylabel:str='Probability',
                                   grid:bool=False,title:str=None,legend:bool=False,savefig:str=None,areame:bool=True) -> pyplot:
    x, y = dataseries(x_), dataseries(y_) # add do-not-sort flag 
    nx, ny = len(x), len(y)
    xval,yval = x.value, y.value
    fig=None
    if ax is None: 
        fig,ax = pyplot.subplots(figsize=figsize)
        if grid: ax.grid()
        C_AREA_PLOTS.reset()
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        if title is not None: ax.set_title(title,fontsize=fontsize)
    else: C_AREA_PLOTS.increment()
    color = COLOR_SEQUENCE[C_AREA_PLOTS.count%10]
    if nx==ny:  # same size case
        u1,v1 = stairs(x)
        u2,v2 = stairs(y)
        ax.fill_betweenx(y=v1,x1=u1,x2=u2,facecolor=color,alpha=alpha)
    else: 
        iqf_x, iqf_y = inverse_quantile_function(x,side='right'), inverse_quantile_function(y,side='right')
        xy = x + y # concatenate the two dataseries # sort happens here again
        ixy = concatenate((zeros((nx,)), ones((ny,))))[xy.index] # 0 for x, 1 for y
        xysv = xy.value_sorted # get data after sorting
        u = diff(xysv) # steps width
        v = iqf_x(xysv) - iqf_y(xysv) # signed steps height
        u_= concatenate(([xysv[0]], xysv[:-1] + u))
        a = asarray([u_[:-1], u_[1:]]).T
        v_ = empty((nx+ny,)) 
        ecdf_p_x, ecdf_p_y = ecdf_p(x), ecdf_p(y)
        vs = v.copy()
        v_[ixy==0] = ecdf_p_x
        v_[ixy==1] = ecdf_p_y
        vs[ixy==0] = -v[ixy==0] # double signed steps height
        b = asarray([v_[:-1], v_[:-1] + vs[:-1]]).T
        for ai,bi in zip(a,b): plot_box(ai,bi,ax=ax,facecolor=color,edgecolor='black')
    plot_ecdf(x,ax=ax)
    plot_ecdf(y,ax=ax)
    if areame: 
        am=areaMe(x,y)
        if nx==ny:ax.text(x.value_sorted[0],0.5,f"area = {'%g'%am}")
        else: ax.text(xysv[0],0.5,f"area = {'%g'%am}")
    if legend: ax.legend()
    if savefig is not None: fig.savefig(savefig)
    return fig,ax

        # h,i=0,0
        # for j,(ind) in enumerate(zip(ixy)):
        #     if ind==0: # x ecdf 
        #         v_[j] = ecdf_p_x[h] 
        #         vs[j] = -v[j] # double signed version of the p differences
        #         h+=1
        #     elif ind==1: # y ecdf
        #         v_[j] = ecdf_p_y[i] 
        #         vs[j] = +v[j] # double signed version of the p differences
        #         i+=1



# def plot(data1, data2, ax=None, alpha=0.3, color='gray', figsize=(12,8), marker='.', lw=1, fontsize='16', xlabel='d1, d2', ylabel='Probability', title=None,legend=True, areait=True, savefig=None):
#     assert type(data1)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
#     assert type(data2)==list, 'datasets need to be python list. Support for numpy arrays to follow.'
#     d1, d2 = Dataset(data1), Dataset(data2)

#     n1,n2 = len(d1),len(d2)

#     x1,y1 = d1.stairs()
#     x2,y2 = d2.stairs()

#     if ax is None:
#         fig, ax = pyplot.subplots(figsize=figsize)
#         ax.grid()

#     ax.plot(x1,y1,lw=lw,marker=marker,label='d1')
#     ax.plot(x2,y2,lw=lw,marker=marker,label='d2')
#     if n1 == n2:
#         ax.fill_betweenx(y1,x1,x2, alpha=alpha, color=color)

#     ax.set_title(title,fontsize=str(int(fontsize)+4))
#     ax.set_xlabel(xlabel,fontsize=fontsize)
#     ax.set_ylabel(ylabel,fontsize=fontsize)
#     if legend:
#         ax.legend()
#     if areait:
#         am = areaMe(data1,data2)
#         xt = (x1[1]+x1[2])/2
#         yt = (y1[0]+y1[1])/2
#         ax.text(xt,yt,f"area = {'%g'%am}")
#     if savefig is not None:
#         fig.savefig(savefig)
#     pass