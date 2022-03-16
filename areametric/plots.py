from matplotlib import pyplot
import numpy

from areametric.dataset import dataset_parser, Dataset, ecdf_, ecdf, DATASET_TYPE

FONTSIZE = 16

def stairs(x:DATASET_TYPE, pseudoinverse:bool=True):
    sort, concat = numpy.sort, numpy.concatenate
    def stepdata(x,y, pseudoi=True): 
        xx,yy = sort(concat((x,x))), sort(concat((y,y))) # sort(2*x), sort(2*y)
        if pseudoi: return xx, concat(([0.],yy[:-1]))
        else: return concat(([xx[0]],xx[:-1])), yy
    x, p = ecdf(x) # x is parsed in ecdf
    x, y = stepdata(x,p, pseudoi=pseudoinverse)
    return x, y

def plot_ecdf(x:DATASET_TYPE,pseudoinverse:bool=True,ax=None,figsize=(12,8),fontsize:int=FONTSIZE):
    fig=[]
    if ax is None:
        fig,ax = pyplot.subplots(figsize=figsize)
        ax.grid()
    x1,y1 = stairs(x,pseudoinverse=pseudoinverse)
    ax.plot(x1,y1)
    ax.scatter(x,[0]*len(x),marker='|',color='red') # https://matplotlib.org/stable/api/markers_api.html
    x2,p=ecdf(x)
    ax.scatter(x2,p,s=10,color='red',zorder=3)
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('p',fontsize=fontsize)
    ax.tick_params(direction='out', grid_alpha=0.5, labelsize='large')
    return fig, ax 

def plot_ecdf_boxed(x: DATASET_TYPE,ax=None,figsize=(12,8),fontsize:int=FONTSIZE):
    fig=[]
    if ax is None:
        fig,ax = pyplot.subplots(figsize=figsize)
        ax.grid()
    x1,y1 = stairs(x,pseudoinverse=True)
    x2,y2 = stairs(x,pseudoinverse=False)
    ax.plot(x1,y1)
    ax.plot(x2,y2)
    ax.fill_between(x=x1,y1=y1,y2=y2,color='gray',alpha=0.1)
    ax.fill_between(x=x2,y1=y1,y2=y2,color='gray',alpha=0.1)
    x3,y3=ecdf_(x)
    ax.plot(x3,y3, color='gray',lw=0.2)
    ax.scatter(x,[0]*len(x),marker='|',color='red')
    ax.set_xlabel('x',fontsize=fontsize)
    ax.set_ylabel('p',fontsize=fontsize)
    ax.tick_params(direction='out', grid_alpha=0.5, labelsize='large') #,colors='#5a5a64'
    return fig, ax 











# def stairs2(x: DatasetType):
#     sort, concat = numpy.sort, numpy.concatenate
#     def stepdata(x,y): 
#         xx,yy = sort(concat((x,x))), sort(concat((y,y))) # sort(2*x), sort(2*y)
#         return concat(([xx[0]],xx[:-1])), yy
#     x, p = ecdf(x) # x is parsed in ecdf
#     x, y = stepdata(x,p)
#     return x, y


# def ecdf(x: list): # x must be python list
#     n = len(x)
#     p = 1/n
#     x.sort() # careful.. sort in place
#     pvalues = list(numpy.linspace(p,1,n))
#     return x, pvalues

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