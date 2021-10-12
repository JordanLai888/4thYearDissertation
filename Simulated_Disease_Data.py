#Creation of the simulated data from the SIR model with added Gaussian Noise to ensure randomised data

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Differential Equations for the SIR model
def DiffEqns(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -(beta*I*S)/N
    dIdt = (beta*I*S)/N - gamma*I
    dRdt = gamma*I
    return dSdt, dIdt, dRdt


#Function to generate the Infected number using the SIR model
def SIRInfectedNumberOutput(PopulationSize, InitialInfected, Beta, Gamma, TimeSeriesLength):
    
    #Initial Conditions
    N = PopulationSize
    I0 = InitialInfected
    R0 = 0
    S0 = N - I0
    y0 = S0, I0, R0 #Initial Conditions vector
   
    #Time series, ie. number of days of data
    t = np.linspace(0,(TimeSeriesLength-1), TimeSeriesLength)
    
    #ODE integrated over time t
    SIRSimulatedValues = odeint(DiffEqns, y0, t, args=(N, Beta, Gamma))
    S, I, R = SIRSimulatedValues.T #Values for the number of S, I and R individuals over time
    NumberInfected = I
    
    #Plot of the predicted S, I and R values over time t 
    fig = plt.figure(facecolor='w')
    SIRPlot = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    SIRPlot.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    SIRPlot.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    SIRPlot.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    SIRPlot.set_xlabel('Time /days')
    SIRPlot.set_ylabel('Number (N)')
    SIRPlot.set_ylim(0,1.1)
    SIRPlot.yaxis.set_tick_params(length=0)
    SIRPlot.xaxis.set_tick_params(length=0)
    SIRPlot.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = SIRPlot.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        SIRPlot.spines[spine].set_visible(False)
    plt.show()
        
    return(NumberInfected)


print(SIRInfectedNumberOutput(10000, 1, 0.5, 0.1, 100))

def SimulatedDiseaseDataWithNoiseAdded(PopulationSize, InitialInfected, Beta, Gamma, TimeSeriesLength):
    
    #Generation of the number of Infected Individuals using the SIR model function above
    SIRSimulation = SIRInfectedNumberOutput(PopulationSize, InitialInfected, Beta, Gamma, TimeSeriesLength)
    
    #Time series, ie. number of days of data
    t = np.linspace(0,(TimeSeriesLength-1), TimeSeriesLength)
    
    #Sampling of the Gaussian Noise
    GaussianSamplingnp = np.random.normal(loc=0, scale=2, size=(len(t)-1)) 
    GaussianSampleAsList = GaussianSamplingnp.tolist()
    GaussianSampleAsList.insert(0,0) 
    
    SimulatedObservations = SIRSimulation + GaussianSampleAsList
    return(SimulatedObservations)
    
InfectedDiseaseData = SimulatedDiseaseDataWithNoiseAdded(10000, 1, 0.5, 0.1, 100)


    
    
    
    

    

    
    
    
