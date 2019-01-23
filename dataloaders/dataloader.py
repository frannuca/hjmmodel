import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import copy as copylib

class HistoricalYieldCurve:
    @staticmethod    
    def __tenor2years(tenor):        
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

    #loads historical data from csv with the following format
    #columns: time,'1m','3m','6m','1y','3y'
    #rows     date,#,#,#,#,#,#,#,...,#
    def __init__(self,path2file,indexcolumn,scalefactor=None):
        self.path2file = path2file
        self.indexcolumn = indexcolumn
        self.scalefactor = scalefactor
        self.data = pd.read_csv(self.path2file,sep=',')        
        self.data.set_index(self.indexcolumn,drop=True,inplace=True)
        scale  = 100
        if scalefactor == None :
            scale = 1.0

        self.data = self.data/scale
        self.tenors = [HistoricalYieldCurve.__tenor2years(x) for x in self.data.columns]
    
        self.diff_rates = pd.DataFrame(np.diff(self.data, axis=0),index=self.data.index[1:],columns=self.data.columns) 
    def plot(self):        
        #self.data.plot()        
        #self.diff_rates.plot() 
        return None   

    def computeVolatilities(self,nfactors):
        self.sigma = np.cov(self.diff_rates.transpose()) * 252.0
        eigval, eigvector = np.linalg.eig(self.sigma)
        index_eigval = list(reversed(np.argsort(eigval)))[0:nfactors]
        self.princ_comp =eigvector[:,index_eigval]
        self.princ_eigval = eigval[index_eigval]
        plt.plot(self.princ_comp, marker='.'), plt.title('Principal components'), plt.xlabel(r'Time $t$')
        plt.figure()        
    
    def computedicretizedvols(self):
        n = self.princ_comp.shape[0]
        aux1 = np.sqrt(np.vstack([self.princ_eigval for s in range(n)]))

        self.vols = np.multiply(aux1,self.princ_comp)       
    
    def computeVolInterpolation(self,mc_steps=51):
        x = np.concatenate(([0],self.tenors))
        self.vol_interpolators = []
        self.vol_interpolators.append(interp1d(x,np.concatenate(([0],self.vols[:,0])),'cubic' ))
        self.vol_interpolators.append(interp1d(x,np.concatenate(([0],self.vols[:,1])),'cubic' ))
        self.vol_interpolators.append(interp1d(x,np.concatenate(([0],self.vols[:,2])),'cubic' ))
        plt.figure()
        plt.plot(self.tenors,self.vols, marker='.'), plt.xlabel(r'Time $t$'), plt.ylabel(r'Volatility $\sigma$'), plt.title('Discretized volatilities')
        self.mc_tenors = np.linspace(self.tenors[0],self.tenors[-1],mc_steps)
        self.mc_vols = np.matrix([[f(t) for t in self.mc_tenors] for f in self.vol_interpolators])
        plt.plot(self.mc_tenors,self.mc_vols.transpose()),plt.xlabel(r'Time $t$'),plt.title('Interpolated volatilities')

    def mdrift(self,tau):
        I = 0

        for f in self.vol_interpolators:
            r,_ = integrate.quad(f,0,tau) * f(tau)
            I += r
        return I
          
    def plotrndrift(self):       
        self.mc_drift = np.array([self.mdrift(tau) for tau in  self.mc_tenors])
        plt.figure()
        plt.plot(self.mc_tenors,self.mc_drift)
    
    def plotFT0_T(self):
        aux1 = self.data.as_matrix()[-1,:].flatten()
        self.curve_spot = np.array(aux1)
        plt.figure()
        plt.plot(self.tenors, self.curve_spot, marker='.'), plt.ylabel('$f(t_0,T)$'), plt.xlabel("$T$")

    def simulateforwards(self,f,proj_timeline):
        #curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline        
        mc_tenors = self.tenors
        mc_vols = self.mc_vols
        len_vols = len(mc_vols)
        len_tenors = len(self.tenors)
        mc_drift = self.mc_drift
        
        
        yield proj_timeline[0],copylib.copy(f)

        for it in range(1,len(proj_timeline)):
            dt = proj_timeline[it]-proj_timeline[it-1]
            t = proj_timeline[it]
            sqrt_dt = np.sqrt(dt)
            fprev = f
            f = copylib.copy(f)
            
            dZ = np.array([np.random.normal() for _ in range(len_vols)])
            for iT in range(len_tenors):
                a = fprev[iT] + mc_drift[iT]*dt
                sum = 0.0
                for iVol, vol in enumerate(np.array(mc_vols)):
                    sum += vol[iT] * dZ[iVol]
                b = sum*sqrt_dt
                if iT+1 < len_tenors:
                    c = (fprev[iT+1]-fprev[iT])/(mc_tenors[iT+1]-mc_tenors[iT])*dt
                else:
                    c = (fprev[iT]-fprev[iT-1])/(mc_tenors[iT]-mc_tenors[iT-1])*dt
                f[iT] = a+b+c
            yield t,f


    def montecarlo(self,seed):
        np.random.seed(seed)
        plt.figure()
        proj_rates = []
        proj_timeline = np.linspace(0,5,51)        
        for i, (t, f) in enumerate(self.simulateforwards(self.curve_spot,proj_timeline)):
            proj_rates.append(f)
        
        proj_rates = np.matrix(proj_rates)
        plt.plot(proj_timeline.transpose(), proj_rates), plt.xlabel(r'Time $t$'), plt.ylabel(r'Rate $f(t,\tau)$');
        plt.title(r'Simulated $f(t,\tau)$ by $t$')
        plt.figure()
        plt.plot(self.mc_tenors, proj_rates.transpose()), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Rate $f(t,\tau)$');
        plt.title(r'Simulated $f(t,\tau)$ by $\tau$')

        
        rshort = []
        rshort.append(proj_rates[0,0])
        I = [0]        
                
        for t in range(0,len(self.mc_tenors)-1):
            dt = self.mc_tenors[t+1]-self.mc_tenors[t]
            I.append(proj_rates[t+1,t+1]*dt)            
            rshort.append(proj_rates[t,t])
            
        I = np.cumsum(I)
        E = np.exp(-1.0 * np.array(I))
        plt.figure()
        plt.plot(self.mc_tenors, rshort), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Short Rate $f(t,\tau)$');
        plt.title(r'Simulated $rshort(t,\tau)$ by $\tau$')
        
        plt.figure()
        plt.plot(self.mc_tenors, E), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Computed Discount $f(t,\tau)$')
        plt.title(r'Simulated $D(t,\tau)$ by $\tau$')

        plt.figure()
        plt.plot(self.mc_tenors, I), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Computed Rate $f(t,\tau)$');
        plt.title(r'Simulated $R(t,\tau)$ by $\tau$')

    def capletpricer(self,K,t_mat,t_exp,notional,n_simulations,n_timesteps):
        proj_timeline = np.linspace(0,t_mat,n_timesteps)
        simulated_forecast_rates = []
        simulated_df = []
        simulated_pvs = []
        pv_convergence_process = []

        for i in range(n_simulations):
            rate_forecast = None
            for t, curve_fwd in self.simulateforwards(self.curve_spot,proj_timeline):
                f_t_0 = interp1d(self.tenors,curve_fwd)