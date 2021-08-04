# Imports of Stars and their data holding objects
import Star as star
import starData as sd
import fourierStarGen as fsg
# Imports of Star data analysis tools
import fourierDecomposition as fd
# Imports libraries for easier use of functions
import numpy as np
import matplotlib.pyplot as plt

# Frequency bounds on Lomb-Scargle Analysis
_LO_ = .4
_HI_ = .5
# The order of the guessed Fourier Function
order = 4

# These variables are used to generate a fake star
times = np.arange(0,20,.01).tolist()
ampCoeff = [11.34,2.72,1.12,4.46,1.87]
phiCoeff = [0,4.56,1.05,4.00,5.13]

# This is a fake star by the name of "Eri". It's at the origin
testingStar = star.Star("Eri",[0,0])
# This adds a data set to "Eri" based of fake data
testingStar.dataSets["fourierGen"] = fsg.fourierStarGen(ampCoeff,phiCoeff,2.213235,times,0,"Cosine")
# This grabs the lightKurve object from fourierGen for graphing
testingCurve = testingStar.dataSets["fourierGen"].lightCurve

# Guess the phase of the fourierGen data set. Zero is initial guess
# TODO:: A better way to create initial guesses and test them
phases = [0,0,0,0]
# Feeds the fourierGen data set to the fourier decomposition function
# TODO:: I need to make this input a type class.
fd.fourierDecomp(testingStar.dataSets["fourierGen"],phaseInfo=[phases,[None,None],True,False],freqInfo=[None,[_LO_,_HI_],False,False],residuals=fd.residuals_cos)

# Makes local copy of results from fourierGen's decomposition
results = testingStar.dataSets["fourierGen"].analysisResults["fourier"]
# This creates a new data set based of the guessed coeff from fourierGen
testingStar.dataSets["fourierFit"] = fsg.fourierStarGen(results["ampCoeff"],results["phiCoeff"],results["period"],times,0,"Cosine")
# Grabs the lightKurve object from fourierFit for graphing
fitCurve = testingStar.dataSets["fourierFit"].lightCurve

# Shows fake data from fourierGen
plt.scatter(testingCurve.time.value,testingCurve.flux.value,color="#1f77b4")
# Shows data from guess based of of fourierGen
plt.plot(fitCurve.time.value,fitCurve.flux.value,color="#ff7f0e")
plt.show()