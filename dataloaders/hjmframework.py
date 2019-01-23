import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import copy as copylib
from .tools import tenor2years,LoadForwardCurves

class HJMFramework:

    @staticmethod
    def computeVolatilities(nfactors,data):
        diff_rates = pd.DataFrame(np.diff(data, axis=0),index=data.index[1:],columns=data.columns)
        sigma = np.cov(diff_rates.transpose()) * 252.0
        eigval, eigvector = np.linalg.eig(sigma)
        index_eigval = list(reversed(np.argsort(eigval)))[0:nfactors]
        princ_comp =eigvector[:,index_eigval]
        princ_eigval = eigval[index_eigval]
        n = princ_comp.shape[0]
        aux1 = np.sqrt(np.vstack([princ_eigval for s in range(n)]))
        vols = np.multiply(aux1,princ_comp)    

        return vols

    def __init__(self,path2file,indexcolumn,scalefactor,nfactors):
        self.tenors,self.data = LoadForwardCurves(path2file,indexcolumn,scalefactor)        
        self.vols = HJMFramework.computeVolatilities(nfactors,self.data)
        self.nfactors = nfactors
    
    def __computeVolInterpolation(self):
        x = np.concatenate(([0],self.tenors))
        self.vol_interpolators = []
        for n in range(self.nfactors):
            if n == 0:
                level = np.mean(np.array(self.vols[:,0]).flatten())
                aux = [level for _ in range(len(x))]
                self.vol_interpolators.append(interp1d(x,aux))
            else:
                self.vol_interpolators.append(interp1d(x,np.concatenate(([self.vols[0,0]],self.vols[:,0])),'cubic' ))                           
        

    def __mdrift(self,T):
        I = 0
        for f in self.vol_interpolators:
            r,_ = integrate.quad(f,0,T) * f(T)
            I += r
        return I

    def __compute_mc_vols_and_drift(self,mc_steps):
        self.__computeVolInterpolation()
        mc_tenors = np.linspace(0,self.tenors[-1],mc_steps)
        mc_vols = np.matrix([[f(t) for t in mc_tenors] for f in self.vol_interpolators])
        mc_drift = np.array([self.__mdrift(tau) for tau in  mc_tenors])

        spot = self.__get_curve_spot()
        f = np.concatenate(([spot[0]],spot)) 
        f_interpolator = interp1d(np.concatenate(([0],self.tenors)) ,f,'cubic')
        mc_forward_curve = np.array([f_interpolator(tau) for tau in  mc_tenors])
        return mc_tenors,mc_vols,mc_drift,mc_forward_curve
   
    def __get_curve_spot(self):
        return self.data.as_matrix()[-1,:].flatten()


    def __run_forward_dynamics(self,proj_time,mc_tenors,mc_vols,mc_drift,mc_forward_curve):                
        len_vols = len(mc_vols)
        len_tenors = len(mc_tenors)         

        yield proj_time[0],copylib.copy(mc_forward_curve)

        for it in range(1, len(proj_time)):
            t = proj_time[it]
            dt = t - proj_time[it-1]
            sqrt_dt = np.sqrt(dt)
            fprev = mc_forward_curve
            mc_forward_curve = copylib.copy(mc_forward_curve)
            dZ = np.array([np.random.normal() for _ in range(len_vols)])
            for iT in range(len_tenors):
                a = fprev[iT] + mc_drift[iT]*dt
                sum = 0.0
                for iVol, vol in enumerate(np.array(mc_vols)):
                    sum += vol[iT] * dZ[iVol]
                b= sum*sqrt_dt

                if iT+1 < len_tenors:
                    c = (fprev[iT+1]-fprev[iT])/(mc_tenors[iT+1]-mc_tenors[iT])*dt
                else:
                    c = (fprev[iT]-fprev[iT-1])/(mc_tenors[iT]-mc_tenors[iT-1])*dt

                mc_forward_curve[iT] = a+b+c

            yield t,mc_forward_curve
    
    def set_montecarlo_parameters(self,seed,timesteps,t_end_years,ntenors):
        np.random.seed(seed)
        self.mc_tenors,self.mc_vols,self.mc_drift,self.mc_forward_curve =  self.__compute_mc_vols_and_drift(ntenors)        
        self.proj_time = np.linspace(0,t_end_years,timesteps).flatten()
        
        
    def run_montecarlo_path(self):
        proj_rates = []
        proj_time = self.proj_time
        mc_tenors = self.mc_tenors
        mc_drift = self.mc_drift
        mc_vols = self.mc_vols
        mc_forward_curve = self.mc_forward_curve

        
        for i, (t, f) in enumerate(self.__run_forward_dynamics(proj_time,mc_tenors,mc_vols,mc_drift,mc_forward_curve)):
            proj_rates.append(f)
        
        columns = [str(int(tn*360)) for tn in mc_tenors]
        proj_rates = pd.DataFrame(np.matrix(proj_rates),index=proj_time,columns=columns)
        return proj_rates
        