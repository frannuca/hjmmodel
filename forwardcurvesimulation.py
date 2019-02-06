import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as copylib
from dataloaders import hjmframework
from dataloaders import tools
from dataloaders import arr

import scipy.integrate as integrate

path2historicalcurve = 'E:/software/hjmmodel/dataloaders/hjm_formatted_1_2_5.csv'
hjmmodel = hjmframework.HJMFramework(path2historicalcurve,indexcolumn='time',scalefactor=100.0,nfactors=3)
hjmmodel.set_montecarlo_parameters(seed = 42,timesteps = 120, t_end_years = 1,ntenors = 100)



path = []

dt = hjmmodel.mc_time[1] - hjmmodel.mc_time[0]
T12 = 1.0
T3 = 3.0/12.0
T6 = 0.5
L12m = []
L3m = []
L6m = []
volumes = []
t=0.9
Nbins = 50
Nsims = 5000
for n in range(1,Nsims):
    print(n)
    cube = hjmmodel.run_montecarlo_path()
     
    iforward = tools.ForwardInterpolator(hjmmodel.mc_time,hjmmodel.mc_tenors,cube.as_matrix())       
    
    
    
    rate = hjmmodel.integrateforward(cube,t,T3)
    L3m.append((np.exp(rate)-1.0)/T3)
    
    rate = hjmmodel.integrateforward(cube,t,T6)
    L6m.append((np.exp(rate)-1.0)/T6)
    
    rate = hjmmodel.integrateforward(cube,t,T12)
    L12m.append((np.exp(rate)-1.0)/T12)

    path.append([L3m[-1],L6m[-1],L12m[-1]])   

path =np.matrix(path)
spreads = (path - np.mean(path,axis=0))*100.0
L3m = np.sort(spreads[:,0])
L6m = np.sort(spreads[:,1])
L12m = np.sort(spreads[:,2])

print("L3m@99={}".format(L3m[int(0.99*Nsims)]))
print("L6m@99={}".format(L6m[int(0.99*Nsims)]))
print("L12m@99={}".format(L12m[int(0.99*Nsims)]))
abins = np.linspace(np.min(L12m),np.max(L12m),Nbins)
plt.hist(L12m,bins=np.array(abins))
plt.title('L12')
plt.figure()

abins = np.linspace(np.min(L3m),np.max(L3m),Nbins)
plt.hist(L3m,bins=np.array(abins))
plt.title('L3')
plt.figure()

abins = np.linspace(np.min(L6m),np.max(L6m),Nbins)
plt.hist(L6m,bins=np.array(abins))
plt.title('L6')

plt.figure()
plt.plot([3,6,12],spreads.transpose())

#plt.plot([3,6,12],np.mean(np.matrix(path),axis=0).transpose()*100.0,linewidth='5',color='k')

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
