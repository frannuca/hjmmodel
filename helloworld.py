import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as copylib
from  dataloaders import dataloader
from  dataloaders import hjmframework


path2historicalcurve = '/home/frannuca/code/python/QuantAndFinancial/heath_jarrow_morton/hjm_formatted_1_2_5.csv'
hjmmodel = hjmframework.HJMFramework(path2historicalcurve,indexcolumn='time',scalefactor=100.0,nfactors=5)
hjmmodel.set_montecarlo_parameters(seed = 42,timesteps = 12, t_end_years = 1,ntenors = 11)



path = []
nt = 12
for n in range(1,25000):
    cube = hjmmodel.run_montecarlo_path()        
    x=np.array(cube.iloc[nt,:]).flatten()     
    path.append(x)
    # if n%1000 == 0:
    #     plt.plot(hjmmodel.mc_tenors, np.array(np.mean(np.matrix(path),axis=0)).flatten(),label=str(n))
    


mu = np.matrix(np.mean(np.matrix(path),axis=0)).transpose()
M = np.matrix(path).transpose() -  mu

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
spreads = np.sort(np.array(M[4,:]).flatten()*100)[::-1]
q = int(0.9997*spreads.size)
print(spreads[q])

#plt.hist(spreads,bins=50,density=True)
abins = np.linspace(np.min(spreads),np.max(spreads),20)
plt.hist(spreads,bins=np.array(abins))
plt.show()
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
