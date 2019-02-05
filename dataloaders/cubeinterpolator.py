import numpy as np
import scipy
import scipy.interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interpn
import numpy as np
import matplotlib.pyplot as plt

class IDWInterpolator:
    def __init__(self,cube):
        self.cube=cube
        
        self.Nvar = cube.shape[1]-1
        self.points = np.matrix(cube[:,0:self.Nvar])#matrix cube with data samples
        self.fx = np.array(cube[:,-1]).flatten()
        
    def interpolateclosestpoints(self,x,n):
        #first locate the time line in the cube
        
        #D = np.linalg.norm(self.points - x,axis=1,ord=1)      
        auxD = np.linalg.norm(self.points - x,axis=1,ord=2)
        auxD2 = auxD*auxD
        sigma = np.std(auxD)
        sigma2 = sigma*sigma
        D = np.exp(-auxD2/sigma2)      
        d = np.argsort(D)[::-1][0:n] 
        D = np.array(D).flatten()[d]
        if D[0]<1e-5:
            return self.fx[d[0]]
        else:
            W = D
            WT = np.sum(W)
            R = self.fx[d]*W/WT
            return np.sum(R)
        

if __name__  =="__main__":
    data = pd.read_csv("C:\\Users\\venus\\code\\github\\hjmmodel\\dataloaders\\points.csv")
    I = IDWInterpolator(data.values)
    
    #0.001589792	0.062131019		0.010497276	0.105548818
    a = 0.02
    b = 0.045
    c = 0.075
    d = 0.10
    x = I.interpolateclosestpoints(np.array([a,b,c,d]),5)
    
    def freal(x,y,z,g):
        return (x*0.2+y*0.6+z*0.33+g*0.05)*500000000

    f = interpolate.LinearNDInterpolator(np.array([data.values[:,0],data.values[:,1], \
                                         data.values[:,2],data.values[:,3]]).transpose(), \
                                         data.values[:,4],  rescale =True)  

    

    fr = freal(a,b,c,d)
    fs = f(a,b,c,d)
    print("computed ({},{},{},{})={}".format(a,b,c,d,x))
    print("expected ({},{},{},{})={}".format(a,b,c,d,fr))
    print("linear ({},{},{},{})={}".format(a,b,c,d,fs))
    print("error={}".format((fr-x)/fr*100))
    print("error linear={}".format((fr-fs)/fr*100))