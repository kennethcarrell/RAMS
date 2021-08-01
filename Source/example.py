import Star as star
import starData as sd
import fourierStarGen as fsg
import fourierDecomposition as fd
import numpy as np
import matplotlib.pyplot as plt

_LO_ = .4
_HI_ = .5
order = 4

times = np.arange(0,20,.01).tolist()
ampCoeff = [11.34,2.72,1.12,4.46,1.87]
phiCoeff = [0,4.56,1.05,4.00,5.13]

testingStar = star.Star("Testing",[0,0])
testingStar.dataSets["fourierGen"] = fsg.fourierStarGen(ampCoeff,phiCoeff,2.213235,times,0,"Cosine")
testingCurve = testingStar.dataSets["fourierGen"].lightCurve

phases = [0,0,0,0]
fd.fourierDecomp(testingStar.dataSets["fourierGen"],phaseInfo=[phases,[None,None],True,False],freqInfo=[None,[_LO_,_HI_],False,False],residuals=fd.residuals_cos)

results = testingStar.dataSets["fourierGen"].analysisResults["fourier"]
testingStar.dataSets["fourierFit"] = fsg.fourierStarGen(results["ampCoeff"],results["phiCoeff"],results["period"],times,0,"Cosine")
fitCurve = testingStar.dataSets["fourierFit"].lightCurve

plt.scatter(testingCurve.time.value,testingCurve.flux.value,color="#1f77b4")
plt.plot(fitCurve.time.value,fitCurve.flux.value,color="#ff7f0e")
plt.show()