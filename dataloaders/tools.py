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