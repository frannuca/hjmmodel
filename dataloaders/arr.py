import numpy as np
import pandas as pd

class ARRCalculator:
    def __init__(self,tenors,wT,totalVolumne):        
        idx = np.argsort(tenors)
        self.tenors = [tenors[n] for n in idx]        
        self.wT = [wT[n] for n in idx]        
        self.totalVolume = totalVolumne
        self.nTenors = len(self.tenors)

    def computeARR(self,nmax,fowardinterpolator,fdVt):
        #compute tenor distribution of volumes
        
        Ni = pd.DataFrame(np.zeros((nmax,self.nTenors)),index=range(0,nmax),columns=self.tenors)
        for itenor in range(self.nTenors):
            tenor = self.tenors[itenor]
            
            for mt in range(0,nmax):
                t = mt*1.0/12
                #homogeneous distribution of the extra volumne across tranches
                Vx = (self.totalVolume+fdVt[mt]/tenor)*self.wT[itenor]/tenor               
                Ni.iloc[mt,itenor] = np.sum(np.array([fowardinterpolator(max([0,t-k/12.0]),tenor/12)*tenor/12 for k in range(0,tenor)])*Vx)

        return Ni

