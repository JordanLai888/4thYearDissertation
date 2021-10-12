import matplotlib.pyplot as plt
import GPy
import numpy as np
import Simulated_Disease_Data as SimulatedData
from scipy.integrate import odeint

InfectedDiseaseData = SimulatedData.InfectedDiseaseData

def DistanceCalc(Beta, Gamma, InitialInfected, PopulationSize , Data):
    N = PopulationSize
    I0 = InitialInfected
    R0 = 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    TimeSeriesLength = len(Data)
    t = np.linspace(0,(TimeSeriesLength-1), TimeSeriesLength)

    RealisationSIR = odeint(SimulatedData.DiffEqns, y0, t, args=(N, Beta, Gamma))
    SRealisation, IRealisation, RRealisation = RealisationSIR.T 
    InfectedRealisation = IRealisation
    
    distance = np.linalg.norm(Data-InfectedRealisation)
    return(distance)



def DistanceValues(betaLower,betaUpper, gammaLower, gammaUpper, increments, InitialInfected, PopulationSize, Data):    
    beta = np.arange(betaLower,(betaUpper + increments), increments)
    gamma = np.arange(gammaLower,(gammaUpper + increments), increments)
    numberValues = len(beta)

    betaMesh, gammaMesh = np.meshgrid(beta, gamma)
    plt.plot(betaMesh, gammaMesh, marker='.', color='k', linestyle='none')

    distanceValues = []
    for i in beta:
        for j in gamma:
            distance = DistanceCalc(i, j, InitialInfected, PopulationSize, Data)
            distanceValues.append(distance)

    distanceValues = np.asarray(distanceValues)
    distanceValues = np.reshape(distanceValues, newshape = (numberValues,numberValues))
    
    ContourDistances = plt.contourf(betaMesh,gammaMesh,np.log(distanceValues), cmap = 'coolwarm')
    plt.colorbar(ContourDistances )
    plt.xlabel('Gamma')
    plt.ylabel('Beta')
    plt.title('L2 Norm Log Distances between simulated and disease data')
    plt.show()
    return(distanceValues)
    
Distances = DistanceValues(0,1,0,1,0.1,1,10000,InfectedDiseaseData)






