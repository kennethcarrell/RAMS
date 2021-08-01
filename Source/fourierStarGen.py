import starData as sd
import numpy as np
import lightkurve as lk
import random

class fourierStarGen(sd.starData):
    def __init__(self,ampCoeff,phiCoeff,period,times,noise=0,fourierType="Cosine"):
        self.ampCoeff = ampCoeff
        self.period = period
        self.omega = (2 * np.pi) / period
        self.phiCoeff = phiCoeff
        self.noise = noise
        self.fourierType = fourierType
        self.flux_mags = list()
        self.times = times
        self.__generateStarCurve()
        self.__translateLightCurveToPeak()
        super().__init__(lk.LightCurve(time=self.times,flux=self.flux_mags))

    def __generateStarCurve(self):
        self.flux_mags = list()
        for time in self.times:
            if (self.fourierType == "Cosine"):
                self.flux_mags.append(self.__iterativeCosineTerms(time))
            elif (self.fourierType == "Sine"):
                self.flux_mags.append(self.__iterativeSineTerms(time))
        if(self.noise!=0):
            self.__generateNoise()
    def __iterativeCosineTerms(self,time):
        total = self.ampCoeff[0]
        for i in range(1,len(self.ampCoeff)):
            total += self.ampCoeff[i] * np.cos(i*self.omega*time+self.phiCoeff[i])
        return total
    def __iterativeSineTerms(self,time):
        total = self.ampCoeff[0]
        for i in range(1,len(self.ampCoeff)):
            total += self.ampCoeff[i] * np.sin(i*self.omega*time+self.phiCoeff[i])
        return total
    def __generateNoise(self):
        random.seed()
        for datum in range(len(self.flux_mags)):
            self.flux_mags[datum] += random.gauss(0,self.noise)
    def __translateLightCurveToPeak(self):
        ## Find the time of the peak flux in the first ~3 days
        translationIndex = np.argmax(self.flux_mags[0:150])
        translationTime = self.times[translationIndex]
        for time in self.times[translationIndex:]:
            time -= translationTime
        self.times = self.times[translationIndex:]
        self.flux_mags = self.flux_mags[translationIndex:]