import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special

def tenor2years(tenor):        
        t = tenor[-1]
        y = tenor[:-1]
        q = float(y)
        if t == 'd':
            return q/360.0
        elif t == 'm':
            return q/12.0
        elif t == 'w':
            return q/(4.0*12.0)
        elif t == 'y':
            return q
        else:
            raise "The provided tenor string {} does not have a correct format".format(tenor)


def LoadForwardCurves(path2file,indexcolumname,scalefactor=None):
    
    data = pd.read_csv(path2file,sep=',')        
    data.set_index(indexcolumname,drop=True,inplace=True)
    scale  = 100
    if scalefactor == None :
        scale = 1.0

    data = data/scale
    tenors = [tenor2years(x) for x in data.columns]
    return tenors,data

def locateboundaries(x, xarr):
     #first locate the time line in the cube
    n = np.argmin(np.abs(xarr-x))
    N = len(xarr)
    xclosest = xarr[n]
    if xclosest < x:
        nlow = n
        nhigh = min([n+1,N-1])
    elif xclosest > x:
        nhigh = n
        nlow = max([n-1,0])
    else:
        nlow  = n
        nhigh = n
    return (nlow,nhigh)

class ForwardInterpolator:
    def __init__(self,mc_time,mc_tenors,cube):
        self.mc_time = mc_time
        self.mc_tenors = mc_tenors
        self.cube=cube
    
    def forward(self,t,T):      
    
        cube = self.cube
        mc_time = self.mc_time
        mc_tenors = self.mc_tenors

        ntlow,nthigh = locateboundaries(t,mc_time)
        nTlow,nThigh = locateboundaries(T,mc_tenors)
        
        x00 = cube[ntlow,nTlow]
        x01 = cube[ntlow,nThigh]
        x10 = cube[nthigh,nTlow]
        x11 = cube[nthigh,nThigh]

        dt = mc_time[nthigh] - mc_time[ntlow]
        dT = mc_tenors[nThigh] - mc_tenors[nTlow]

        tnorm = (t-mc_time[ntlow])/dt
        if dt == 0:
            tnorm = 0
        Tnorm = (T - mc_tenors[nTlow])/dT
        
        return x00*(1-tnorm)*(1-Tnorm)+x10*tnorm*(1-Tnorm)+x01*(1-tnorm)*Tnorm+x11*tnorm*Tnorm
