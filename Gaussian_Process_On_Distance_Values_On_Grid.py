import matplotlib.pyplot as plt
import GPy
import numpy as np
from scipy.stats.distributions import norm
import Simulated_Disease_Data as SimulatedData
from pyDOE import *
from mpl_toolkits import mplot3d
import Distance_Calculation as DistCalc
%matplotlib inline

InfectedDiseaseData = SimulatedData.InfectedDiseaseData


def BetaValues(betaLower, betaUpper, increments):
    beta = np.arange(betaLower,(betaUpper + increments), increments)
    return(beta)

def GammaValues(gammaLower, gammaUpper, increments):
    gamma = np.arange(gammaLower,(gammaUpper + increments), increments)
    return(gamma)

Beta = BetaValues(0,1,0.1)
Gamma = GammaValues(0,1,0.1)
betaMesh, gammaMesh = np.meshgrid(Beta, Gamma)

grid = np.array([betaMesh.flatten(), gammaMesh.flatten()]).T
print(grid)
print(grid.shape)
print(Beta)
print(betaMesh)

distancesGrid = []
for i in Beta:
    for j in Gamma:
        distance = DistCalc.DistanceCalc(i, j, 1, 10000, InfectedDiseaseData)
        distancesGrid.append(distance)
#print(distancesGrid)

distancesGrid = np.asarray(distancesGrid)[:,None]
#print(distancesGrid)
#print(distancesGrid.shape)

#kernel = GPy.kern.Exponential(2, ARD=True)
kernel = GPy.kern.RBF(2,ARD = True)
GPyGridModel = GPy.models.GPRegression(grid, np.log(distancesGrid), kernel)
GPyGridModel.Gaussian_noise.variance.fix(0.001)
GPyGridModel.optimize_restarts(20)
GPyGridModel.optimize()
print(GPyGridModel)
GPyGridModel.plot()


betaplt = np.linspace(0,1,100)
gammaplt = np.linspace(0,1,100)
betaMeshplt, gammaMeshplt = np.meshgrid(betaplt, gammaplt)


xplt = np.array([betaMeshplt.flatten(), gammaMeshplt.flatten()]).T
print(xplt)
print(xplt.shape)


predValues,_ = GPyGridModel.predict(xplt)
print(predValues,_)
print(predValues.shape)
predValuesMatrix = predValues.reshape((100,100))
print(predValuesMatrix.shape)

ContourPredict = plt.contourf(betaplt,gammaplt,predValuesMatrix, cmap = 'coolwarm')
plt.colorbar(ContourPredict )
plt.xlabel('Gamma')
plt.ylabel('Beta')
plt.title('Contour plot for the Gaussian Process trained over grid values of gamma and beta')
plt.show()

print(GPyGridModel)




