#Find distance between first two peaks and translate both formula to that as start.

import re
from lightkurve import search_tesscut
from astropy.table import Table
from astropy.io import fits
from scipy.optimize import least_squares
from fourierStar import fourierStar
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import warnings
import lightkurve as lk
import timeit
warnings.filterwarnings("ignore")

### CONSTANTS FOR LATER
_THRESH_ = 3
_LO_ = .4
_HI_ = .5
order = 4

####
#### Here is the fourier series where phase and amplitude need to be guessed. Frequency and time are given.
####
def fourier_series_sin(f, x, pars, order):
    sum0 = pars[0]
    for i in range(order):
        sum0 += pars[i*2+1]*np.sin((i+1)*2.0*np.pi*f*x + pars[i*2+2])
    return sum0
def fourier_series_cos(f, x, pars, order):
    sum0 = pars[0]
    for i in range(order):
        sum0 += pars[i*2+1]*np.cos((i+1)*2.0*np.pi*f*x + pars[i*2+2])
    return sum0

####
#### Function to give to least-squares fitting
####
def residuals_sin(pars, f, x, y, order):
    return y - fourier_series_sin(f, x, pars, order)
def residuals_cos(pars, f, x, y, order):
    return y - fourier_series_cos(f, x, pars, order)


def findAmplitudeSumOfSquares(amplitudes, results):
    sumOfSquares = (amplitudes[0] - results.x[0])**2
    for i in range(len(amplitudes)-1):
        sumOfSquares += (amplitudes[i+1] - results.x[i*2+1])**2
    return sumOfSquares
def findPhaseSumOfSquares(phases, results):
    # finds n2pi above and below phase
    phasesToCheck = list()
    for phase in phases:
        phasesToCheck.append(list())
        for i in np.arange(-2,2,1):
            phasesToCheck[-1].append(phase+i*np.pi)
    #finds the sum of squares that is least for 2pi translations of the known phase
    sumOfSquares = 0#(phases[0] - results.x[0])**2
    for i in range(len(phases)-1):
        diffSquares = list()
        for phase in phasesToCheck[i]:
            diffSquares.append((phase-results.x[i*2+2])**2)
        sumOfSquares += min(diffSquares)
    return sumOfSquares

def findSumOfSquaresDataPoints(mags1,mags2):
    sumOfSquaresAvg = 0
    lowestMags = None
    if (len(mags1) <= len(mags2)):
        lowestMags = mags1
    else:
        lowestMags = mags2
    for i in range(len(lowestMags)):
        sumOfSquaresAvg += (mags1[i]-mags2[i])**2
    return sumOfSquaresAvg/len(lowestMags)


# Star plotted is UY Eri
times = np.arange(0,20,.01).tolist()
amplitudes = [11.34,2.72,1.12,4.46,1.87]
phiValues = [0,4.56,1.05,4.00,5.13]
star = fourierStar(amplitudes=amplitudes, period=2.213235, phiValues=phiValues, dataNoise=.5, times=times)
lc = star.getLightCurve()


# Decompose the lc light curve to approximate for the Fourier coefficients
phasesRan = np.arange(0.01,2*np.pi,.1).tolist()
sineDecomposition = list()
cosineDecomposition = list()
for phase in phasesRan:
    phases = [phase,phase,phase,phase]
    sineDecomposition.append(star.doFourierDecomp(residuals_sin,phaseInfo=[phases,[None,None],True,False],freqInfo=[None,[_LO_,_HI_],False,False]))
    cosineDecomposition.append(star.doFourierDecomp(residuals_cos,phaseInfo=[phases,[None,None],True,False],freqInfo=[None,[_LO_,_HI_],False,False]))

squaresASin = list()
squaresPSin = list()
squaresSin = list()
squaresACos = list()
squaresPCos = list()
squaresCos = list()

for result in sineDecomposition:
    squaresASin.append(findAmplitudeSumOfSquares(amplitudes,result))
    squaresPSin.append(findPhaseSumOfSquares(phiValues,result))
    # Here are the new coefficients
    amplitudesNew = [result.x[0],result.x[1],result.x[3],result.x[5],result.x[7]]
#results = cosineDecomposition[minimumSumPhaseIndex]
    phiValuesNew = [0,result.x[2],result.x[4],result.x[6],result.x[8]]
    newLC = fourierStar(amplitudes=amplitudesNew,period = 2.213235,phiValues=phiValuesNew,times=times,fourierType="Sine")
    squaresSin.append(findSumOfSquaresDataPoints(newLC.flux_mags,star.flux_mags))

for result in cosineDecomposition:
    squaresACos.append(findAmplitudeSumOfSquares(amplitudes,result))
    squaresPCos.append(findPhaseSumOfSquares(phiValues,result))



#minimumSumAmpIndex = squaresACos.index(min(squaresACos))
#minimumSumPhaseIndex = squaresPCos.index(min(squaresPCos))
#minThingy = squaresACos.index(min(squaresACos))
minThingy = squaresSin.index(min(squaresSin))

results = cosineDecomposition[minThingy]

# Here are the new coefficients
amplitudesNew = [results.x[0],results.x[1],results.x[3],results.x[5],results.x[7]]
#results = cosineDecomposition[minimumSumPhaseIndex]
phiValuesNew = [0,results.x[2],results.x[4],results.x[6],results.x[8]]
periodNew = 1/star.decomp_lspg_f1


# The new coefficients are used to generate a new star and get its light curve
newLC = fourierStar(amplitudes=amplitudesNew,period = periodNew,phiValues=phiValuesNew,times=times,fourierType="Cosine").getLightCurve()
# I am plotting the two light curves
fig, ax = plt.subplots(2)
lc.scatter(ax = ax[0])
ax[0].scatter(lc.time.value,lc.flux.value,color="#1f77b4")
ax[0].plot(newLC.time.value,newLC.flux.value,color="#ff7f0e")
ax[1].scatter(phasesRan,squaresASin,label="Sin Amplitudes")
ax[1].scatter(phasesRan,squaresPSin,label="Sin Phases")
#ax[1].scatter(phasesRan,squaresACos,label="Cos Amplitudes")
#ax[1].scatter(phasesRan,squaresPCos,label="Cos Phases")
#ax[1].scatter(phasesRan,squaresSin)
ax[0].set_title("Best Light Curve")
ax[1].set_title("Least Squared Distances")
ax[1].set_xlabel("Phases Attempted")
ax[1].legend()
fig.tight_layout()
plt.show()