#TODO:: The guesses are passing a list of every guess, and not just one guess!





# Tool for generating plausable low resolution light curves and finding their period

from lightkurve import time
import matplotlib.pyplot as plt
import numpy as np
import random
import lightkurve as lk
from scipy.optimize import least_squares
import statsmodels.api as sm


class fourierStar(object):
    def __init__(self,amplitudes=[], period=0, phiValues=[], dataNoise = 0, times=[], flux_mags=[], data=[[],[]],fourierType = "Cosine"):
        # Generates data from coefficients and times range
        self.amplitudes = amplitudes
        self.period = period
        self.omega = (2 * np.pi) / period
        self.phiValues = phiValues
        self.dataNoise = dataNoise
        self.fourierType = fourierType
        # Imports data from existing data points
        self.times = times
        self.flux_mags = flux_mags
        # creates fourier star curve if the amplitude, period, phiValues, and time domain are given
        if(period!=0):
            self.generateStarCurve()
        # creates light curve from data, translates it to nearest maximum, and rewrites to data
        self.setLightCurve(self.times,self.flux_mags)
        self.__translateLightCurveToPeak()
        self.setLightCurve(self.times,self.flux_mags)
        # Fourier Decomposition Variables
        self.decomp_order = None
        self.decomp_lspg = None
        self.decomp_lspg_f1 = None
        self.decomp_guess = None

    def setLightCurve(self,times,flux_mags):
        self.lightCurve = lk.LightCurve(time=times,flux=flux_mags)

    # Creates UY Starcurve from Fourier analysis in Simon and Lee.
    def generateStarCurve(self):
        self.flux_mags = list()
        for time in self.times:
            if (self.fourierType == "Cosine"):
                self.flux_mags.append(self.__iterativeCosineTerms(time))
            elif (self.fourierType == "Sine"):
                self.flux_mags.append(self.__iterativeSineTerms(time))
        if(self.dataNoise!=0):
            self.__generateNoise()
    def __iterativeCosineTerms(self,time):
        total = self.amplitudes[0]
        for i in range(1,len(self.amplitudes)):
            total += self.amplitudes[i] * np.cos(i*self.omega*time+self.phiValues[i])
        return total
    def __iterativeSineTerms(self,time):
        total = self.amplitudes[0]
        for i in range(1,len(self.amplitudes)):
            total += self.amplitudes[i] * np.sin(i*self.omega*time+self.phiValues[i])
        return total
    def __generateNoise(self):
        random.seed()
        for datum in range(len(self.flux_mags)):
            self.flux_mags[datum] += random.gauss(0,self.dataNoise)

    # I want to try and do this after the stuff is guessed, so it will be in phase better
    # This translates data to the first large peak. Makes fourier decomposition easier among other things
    def __translateLightCurveToPeak(self):
        ## Find the time of the peak flux in the first ~3 days
        translationIndex = np.argmax(self.lightCurve.flux.value[0:150])
        translationTime = self.lightCurve.time.value[translationIndex]
        self.times = self.lightCurve.time.value[translationIndex:] - translationTime
        self.flux_mags = self.lightCurve.flux.value[translationIndex:]

    #Info=[variablesList,Bounds2DList,guessVariable?,guessBounds?]
    def doFourierDecomp(self,residuals,order=4,ampInfo=[None,[None,None],True,False],
                        phaseInfo=[None,[None,None],True,False],
                        freqInfo=[None,[None,None],False,False],
                        timeInfo=[None,[None,None],False,False]):
        self.decomp_order = order
        self.__createInitialGuessesOrArgs(ampInfo,phaseInfo,freqInfo,timeInfo)
        self.__findBounds()
        return self.__leastSquaresDecomposition(residuals)



    # This either generates the initial guesses, or it sets the args for the least squares function
    def __createInitialGuessesOrArgs(self,ampInfo,phaseInfo,freqInfo,timeInfo):
        self.fd_guesses = list()
        self.fd_args = list()
        self.fd_ampGuesses,self.fd_ampBounds = list(),list()
        self.fd_phaseGuesses,self.fd_phaseBounds = list(),list()
        self.fd_freqGuesses,self.fd_freqBounds = list(),list()
        self.fd_timeGuesses,self.fd_timeBounds = list(),list()
        emptyFunction = lambda empty: None
        self.__sortGuessesAndKnowns(ampInfo,self.fd_ampGuesses,self.fd_ampBounds,emptyFunction,self.__ampIgnorantGuess)
        self.__sortGuessesAndKnowns(phaseInfo,self.fd_phaseGuesses,self.fd_phaseBounds,emptyFunction,self.__phaseIgnorantGuess)
        self.__sortGuessesAndKnowns(freqInfo,self.fd_freqGuesses,self.fd_freqBounds,self.__freqKnownBounds,self.__freqIgnorantGuess)
        self.__sortGuessesAndKnowns(timeInfo,self.fd_timeGuesses,self.fd_timeBounds,emptyFunction,self.__timeIgnorantGuess)
        self.__organizeGuessingList(ampInfo[2],phaseInfo[2],freqInfo[2],timeInfo[2])
        # Finishes generating the argument list to be passed to Least Squares
        self.fd_args.append(self.times)
        self.fd_args.append(self.flux_mags)
        self.fd_args.append(self.decomp_order)
        
    # This structure works for all info types
    def __sortGuessesAndKnowns(self,infoVar,infoVarGuessList,infoVarBoundList,setBoundKnownFunction,setIgnorantFunction):
        if infoVar[0]!=None:
            infoVarGuessList.extend(infoVar[0]) if infoVar[2] else self.fd_args.append(self.infoValue[0])
        elif infoVar[1][0]!=None:
            infoVarBoundList.extend(infoVar[1]) if infoVar[3] else setBoundKnownFunction(infoVar[1])
        else:
            setIgnorantFunction()

    def __ampIgnorantGuess(self):
        a0 = np.mean(self.flux_mags)
        a1 = (np.max(self.flux_mags) - np.min(self.flux_mags)) / 2.0
        self.fd_ampGuesses = [a0]
        for i in range(self.decomp_order):
            self.fd_ampGuesses.append(a1/(2*(i+1)))
    def __phaseIgnorantGuess(self):
        for i in range(self.decomp_order):
            self.fd_phaseGuesses.append(0)
    def __freqIgnorantGuess(self):
        pass
    def __timeIgnorantGuess(self):
        self.fd_timeGuesses=[]
        for i in range(self.decomp_order):
            self.fd_timeGuesses.append(0)

    def __freqKnownBounds(self,freqBounds):
        self.__lombScargleAnalysis(freqBounds)
        print("Works Here")
        self.fd_args.append(self.decomp_lspg_f1)

    # Properly Fills the guessing list to be passed to Least Squares
    def __organizeGuessingList(self,ampGuess,phaseGuess,freqGuess,timeGuess):
        if ampGuess:
            self.fd_guesses.append(self.fd_ampGuesses[0])
        for i in range(self.decomp_order):
            if ampGuess:
                self.fd_guesses.append(self.fd_ampGuesses[i+1])
            if phaseGuess:
                self.fd_guesses.append(self.fd_phaseGuesses[i])
            if freqGuess:
                self.fd_guesses.append(self.fd_freqGuesses[i])
            if timeGuess:
                self.fd_guesses.append(self.fd_timeGuesses[i])

    ## Lomb-Scargle analysis of full dataset to get the frequency
    def __lombScargleAnalysis(self,freqBounds):
        lspg = self.lightCurve.to_periodogram(minimum_frequency=freqBounds[0],maximum_frequency=freqBounds[1],oversample_factor=100)
        ## Highest peak in LS analysis
        self.decomp_lspg_f1 = lspg.frequency_at_max_power.value
    
    # This does not change the way iniana.py guessed amplitudes
    def __findBounds(self):
        blo = [0]
        bhi = [np.inf]
        for i in range(self.decomp_order):
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
        print(self.fd_args)
        return least_squares(residuals, self.fd_guesses, bounds=self.decomp_bnds, args=self.fd_args)

    def getLightCurve(self):
        return self.lightCurve
    