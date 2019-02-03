import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as copylib
from  dataloaders import hjmframework
from dataloaders import tools
from dataloaders import arr

import scipy.integrate as integrate

path2historicalcurve = 'C:/Users/venus/code/github/hjmmodel/dataloaders/hjm_formatted_1_2_5.csv'
hjmmodel = hjmframework.HJMFramework(path2historicalcurve,indexcolumn='time',scalefactor=100.0,nfactors=3)
hjmmodel.set_montecarlo_parameters(seed = 42,timesteps = 120, t_end_years = 1,ntenors = 100)



path = []
nt = 11
dt = hjmmodel.mc_time[1] - hjmmodel.mc_time[0]
T12 = int(1.0/dt)
T3 = int(0.25/dt)
T6 = int(0.5/dt)
L12m = []
L3m = []
L6m = []
volumes = []
for n in range(1,300):
    print(n)
    cube = hjmmodel.run_montecarlo_path()
     
    iforward = tools.ForwardInterpolator(hjmmodel.mc_time,hjmmodel.mc_tenors,cube.as_matrix())
   
    def ff(t,T):
        ivals = lambda Tx: iforward.forward(t,Tx)
        ee,_ = integrate.quadpack.quad(ivals,0,T)
        x = (np.exp(ee)-1.0)/T
        return x

    oarr = arr.ARRCalculator([1,3,6],[0.40,0.2,0.2],500e6)   

    frame = oarr.computeARR(12,ff,[0 for _ in range(12)])
    #frame.to_csv("C:/Users/venus/temp/NII.csv")
    volumes.append(frame.sum(axis=1).values/1e6)
    
    t=0.25
    
    ivals = lambda T: iforward.forward(t,T)
    ee12,_ = integrate.quadpack.quad(ivals,0,1.0)
    L12m.append((np.exp(ee12)-1.0))
    ee3m,_ = integrate.quadpack.quad(ivals,0,0.25)
    L3m.append((np.exp(ee3m)-1.0)/0.25)
    ee6m,_ = integrate.quadpack.quad(ivals,0,0.5)
    L6m.append((np.exp(ee6m)-1.0)/0.5)
    
    x=np.array(cube.iloc[nt,:]).flatten()     
    path.append(x)
    # if n%1000 == 0:
    #     plt.plot(hjmmodel.mc_tenors, np.array(np.mean(np.matrix(path),axis=0)).flatten(),label=str(n))
    

abins = np.linspace(np.min(L12m),np.max(L12m),10)
plt.hist(L12m,bins=np.array(abins))
plt.figure()

abins = np.linspace(np.min(L3m),np.max(L3m),10)
plt.hist(L3m,bins=np.array(abins))
plt.figure()
abins = np.linspace(np.min(L6m),np.max(L6m),10)
plt.hist(L6m,bins=np.array(abins))

plt.figure()
plt.plot(np.matrix(volumes).transpose())


plt.figure()

dV = np.array(np.sum(np.matrix(volumes),axis=1)).flatten()
print(dV)
abins = np.linspace(np.min(dV),np.max(dV),20)
plt.hist(dV,bins=np.array(abins))

plt.show()


#mu = np.matrix(np.mean(np.matrix(path),axis=0)).transpose()
#M = np.matrix(path).transpose() -  mu

#plt.plot(hjmmodel.mc_tenors, M)
#plt.plot(hjmmodel.mc_tenors,mu)
#plt.title("t {}".format(cube.index[10]))
#plt.title("tenor {}".format(hjmmodel.tenors[0]))

#plt.plot(hjmmodel.mc_tenors, np.array(mu).flatten(),linewidth=4,color='b',marker='x',label='final')
#plt.plot(hjmmodel.mc_tenors, cube.iloc[0,:],linewidth=4,color = 'r', marker='o',label='f(0,tenor)')

# cube = hjmmodel.run_montecarlo_path()
# cube.plot()
# plt.title("time={}m".format(hjmmodel.proj_time[nt]*12))
# plt.legend()

#plt.figure()
#spreads = np.sort(np.array(M[4,:]).flatten()*100)[::-1]
#q = int(0.9997*spreads.size)
#print(spreads[q])

#plt.hist(spreads,bins=50,density=True)
#abins = np.linspace(np.min(spreads),np.max(spreads),20)
#plt.hist(spreads,bins=np.array(abins))
#plt.show()
#histcurve = HistoricalYieldCurve(path2historicalcurve,'time',100.0)

#histcurve.computeVolatilities(3)
#histcurve.computedicretizedvols()
#histcurve.computeVolInterpolation()
#histcurve.plotrndrift()
#histcurve.plotFT0_T()
#histcurve.plot()
#for see in range(1):
#    histcurve.montecarlo(see)



#plt.show()
