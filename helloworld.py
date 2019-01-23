import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as copylib
from  dataloaders import dataloader
from  dataloaders import hjmframework


path2historicalcurve = '/home/frannuca/code/python/QuantAndFinancial/heath_jarrow_morton/hjm_formatted.csv'
hjmmodel = hjmframework.HJMFramework(path2historicalcurve,'time',100.0,3)
hjmmodel.set_montecarlo_parameters(42,120,1,12)
path = hjmmodel.run_montecarlo_path()
path.plot()
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
