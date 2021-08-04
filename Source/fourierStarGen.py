import starData as sd
import numpy as np
import lightkurve as lk
import random

# Subclass of starData that creates data based off of input coefficients
class fourierStarGen(sd.starData):
    def __init__(self,ampCoeff,phiCoeff,period,times,noise=0,fourierType="Cosine"):
        # Coefficients of fourier expansion
        self.ampCoeff = ampCoeff
        self.period = period
        self.omega = (2 * np.pi) / period
        self.phiCoeff = phiCoeff
        # Level of gaussian randomness to add to datapoints
        self.noise = noise
        # Sets fourier series as its Sine or Cosine form
        self.fourierType = fourierType
        # Times and amplitudes to feed lightKurve.
        # TODO:: Not technically necissary. Removal results in less memory, but less clarity in code.
        self.flux_mags = list()
        self.times = times
        # Creates data points and sets them
        self.__generateStarCurve()
        self.__translateLightCurveToPeak()
        # Feads the lightCurve to the parent class's variables
        super().__init__(lk.LightCurve(time=self.times,flux=self.flux_mags))

    # Generates the light intensity for each time point given using fourier series
    def __generateStarCurve(self):
        self.flux_mags = list()
        for time in self.times:
            #Processes differently if using Cosine or Sine expansion
            if (self.fourierType == "Cosine"):
                self.flux_mags.append(self.__iterativeCosineTerms(time))
            elif (self.fourierType == "Sine"):
                self.flux_mags.append(self.__iterativeSineTerms(time))
        # If the random noise is given, generate it. Otherwise don't waste time
        if(self.noise!=0):
            self.__generateNoise()
    # Makes code more readable in __generateStarCurve. i cosine term
    def __iterativeCosineTerms(self,time):
        total = self.ampCoeff[0]
        for i in range(1,len(self.ampCoeff)):
            total += self.ampCoeff[i] * np.cos(i*self.omega*time+self.phiCoeff[i])
        return total
    # Makes code more readable in __generateStarCurve. i sine term
    def __iterativeSineTerms(self,time):
        total = self.ampCoeff[0]
        for i in range(1,len(self.ampCoeff)):
            total += self.ampCoeff[i] * np.sin(i*self.omega*time+self.phiCoeff[i])
        return total
    # Moves intensity up or down by guassian random distribution
    def __generateNoise(self):
        random.seed()
        for datum in range(len(self.flux_mags)):
            self.flux_mags[datum] += random.gauss(0,self.noise)
    # Moves initial time to the top of the peak
    def __translateLightCurveToPeak(self):
        ## Find the time of the peak flux in the first ~3 days
        translationIndex = np.argmax(self.flux_mags[0:150])
        translationTime = self.times[translationIndex]
        for time in self.times[translationIndex:]:
            time -= translationTime
        self.times = self.times[translationIndex:]
        self.flux_mags = self.flux_mags[translationIndex:]