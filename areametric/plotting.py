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

import numpy
from numpy import (ndarray, sort, concatenate)

from matplotlib import pyplot

from .areametric import dataseries
from .methods import (ecdf,ecdf_)

FONTSIZE = 16
FIGSIZE = (12,8)

class ECDF_BOXED():
    count=0
    def __init__(self): None
    def increment(self): ECDF_BOXED.count+=1
    def reset(self): ECDF_BOXED.count=0

C_ECDF_BOXED = ECDF_BOXED()


# https://matplotlib.org/stable/users/prev_whats_new/dflt_style_changes.html
# cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

COLOR_SEQUENCE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def stairs(x:ndarray, above:bool=True):
    def stepdata(x,y): 
        xx,yy = sort(concatenate((x,x))), sort(concatenate((y,y))) # sort(2*x), sort(2*y)
        if above: return xx, concatenate(([0.],yy[:-1]))
        else: return concatenate(([xx[0]],xx[:-1])), yy
    x,p = ecdf(x) # x is parsed in ecdf
    x,y = stepdata(x,p)
    return x, y

def plot_ecdf(x_:ndarray,ax=None,figsize=FIGSIZE,fontsize:int=FONTSIZE,grid=False):
    x = dataseries(x_).value
    fig=[]
    if ax is None:fig,ax = pyplot.subplots(figsize=figsize)
    if grid: ax.grid()
    x1,y1 = stairs(x)
    ax.plot(x1,y1)
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
        C_ECDF_BOXED.reset() 
    else: C_ECDF_BOXED.increment() # increment counter to move on to the next color in the sequence. 
    color = COLOR_SEQUENCE[C_ECDF_BOXED.count%10]
    if grid: ax.grid()
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
