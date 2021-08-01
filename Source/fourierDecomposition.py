import numpy as np
from scipy.optimize import least_squares



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
def residuals_sin(pars, f, x, y, order):
    return y - fourier_series_sin(f, x, pars, order)
def residuals_cos(pars, f, x, y, order):
    return y - fourier_series_cos(f, x, pars, order)


class fourierDecomp(object):
    def __init__(self,starData,ampInfo=[None,[None,None],True,False],
                        phaseInfo=[None,[None,None],True,False],
                        freqInfo=[None,[None,None],False,False],
                        timeInfo=[None,[None,None],False,False],
                        residuals=None,order=4):
        self.order = order
        self.lightCurve = starData.lightCurve
        self.flux_mags = starData.lightCurve.time.value
        self.times = starData.lightCurve.flux.value
        self.guesses = list()
        self.args = list()
        self.ampGuesses,self.ampBounds = list(),list()
        self.phaseGuesses,self.phaseBounds = list(),list()
        self.freqGuesses,self.freqBounds = list(),list()
        self.timeGuesses,self.timeBounds = list(),list()
        self.__createInitialGuessesOrArgs(starData,ampInfo,phaseInfo,freqInfo,timeInfo)
        self.__findBounds()
        self.__storeResultsInStarData(starData,residuals)
        

    # This either generates the initial guesses, or it sets the args for the least squares function
    def __createInitialGuessesOrArgs(self,starData,ampInfo,phaseInfo,freqInfo,timeInfo):
        emptyFunction = lambda empty: None
        self.__sortGuessesAndKnowns(ampInfo,self.ampGuesses,self.ampBounds,emptyFunction,self.__ampIgnorantGuess)
        self.__sortGuessesAndKnowns(phaseInfo,self.phaseGuesses,self.phaseBounds,emptyFunction,self.__phaseIgnorantGuess)
        self.__sortGuessesAndKnowns(freqInfo,self.freqGuesses,self.freqBounds,self.__freqKnownBounds,self.__freqIgnorantGuess)
        self.__sortGuessesAndKnowns(timeInfo,self.timeGuesses,self.timeBounds,emptyFunction,self.__timeIgnorantGuess)
        self.__organizeGuessingList(ampInfo[2],phaseInfo[2],freqInfo[2],timeInfo[2])
        # Finishes generating the argument list to be passed to Least Squares
        self.args.append(self.flux_mags)
        self.args.append(self.times)
        self.args.append(self.order)
        
    # This structure works for all info types
    def __sortGuessesAndKnowns(self,infoVar,infoVarGuessList,infoVarBoundList,setBoundKnownFunction,setIgnorantFunction):
        if infoVar[0]!=None:
            infoVarGuessList.extend(infoVar[0]) if infoVar[2] else self.args.extend(infoVar[0])
        elif infoVar[1][0]!=None:
            infoVarBoundList.extend(infoVar[1]) if infoVar[3] else setBoundKnownFunction(infoVar[1])
        else:
            setIgnorantFunction()

    def __ampIgnorantGuess(self):
        a0 = np.mean(self.flux_mags)
        a1 = (np.max(self.flux_mags) - np.min(self.flux_mags)) / 2.0
        self.ampGuesses = [a0]
        for i in range(self.order):
            self.ampGuesses.append(a1/(2*(i+1)))
    def __phaseIgnorantGuess(self):
        for i in range(self.order):
            self.phaseGuesses.append(0)
    def __freqIgnorantGuess(self):
        pass
    def __timeIgnorantGuess(self):
        self.timeGuesses=[]
        for i in range(self.order):
            self.timeGuesses.append(0)

    def __freqKnownBounds(self,freqBounds):
        self.__lombScargleAnalysis(freqBounds)
        self.args.append(self.decomp_lspg_f1)

    # Properly Fills the guessing list to be passed to Least Squares
    def __organizeGuessingList(self,ampGuess,phaseGuess,freqGuess,timeGuess):
        if ampGuess:
            self.guesses.append(self.ampGuesses[0])
        for i in range(self.order):
            if ampGuess:
                self.guesses.append(self.ampGuesses[i+1])
            if phaseGuess:
                self.guesses.append(self.phaseGuesses[i])
            if freqGuess:
                self.guesses.append(self.freqGuesses[i])
            if timeGuess:
                self.guesses.append(self.timeGuesses[i])

    ## Lomb-Scargle analysis of full dataset to get the frequency
    def __lombScargleAnalysis(self,freqBounds):
        lspg = self.lightCurve.to_periodogram(minimum_frequency=freqBounds[0],maximum_frequency=freqBounds[1],oversample_factor=100)
        ## Highest peak in LS analysis
        self.decomp_lspg_f1 = lspg.frequency_at_max_power.value
    
    # This does not change the way iniana.py guessed amplitudes
    def __findBounds(self):
        blo = [0]
        bhi = [np.inf]
        for i in range(self.order):
            blo.append(0)
            bhi.append(np.inf)
            blo.append(0)
            bhi.append(2.0*np.pi)
        #bnds = ([0, 0,0,0, 0,0,0, 0,0,0],[np.inf, np.inf,np.inf,2.0*np.pi, np.inf,np.inf,2.0*np.pi, np.inf,np.inf,2.0*np.pi])
        self.decomp_bnds = (blo,bhi)
    # Given guesses, this tries to find a solution to the fourier decomposition
    def __leastSquaresDecomposition(self,residuals):
        ## Fit with a least-squares fitter
        #results = least_squares(residuals, guess, bounds=bnds, args=(times, mags))
        return least_squares(residuals, self.guesses, bounds=self.decomp_bnds, args=self.args)

    def __storeResultsInStarData(self,starData,residuals):
        results = self.__leastSquaresDecomposition(residuals)
        starData.analysisResults["fourier"] = dict()
        starData.analysisResults["fourier"]["ampCoeff"] = [results.x[0],results.x[1],results.x[3],results.x[5],results.x[7]]
        starData.analysisResults["fourier"]["phiCoeff"] = [0,results.x[2],results.x[4],results.x[6],results.x[8]]
        starData.analysisResults["fourier"]["period"] = 1/self.decomp_lspg_f1 