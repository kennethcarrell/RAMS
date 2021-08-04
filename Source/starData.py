# This is object holds the lightKurve light curve for a star from some
# data set. There can be as many of these inserted in the star object
# as there are independent data sets.
# 
# It also stores the results of different types of analysis in a dict.
# The dict holds other dicts, each of the analysis method done on the
# data set. This allows the dynamic storage of the results from multiple 
# analysis techniques.
#
# This class is designed to be inherited by new sub-classes that include
# traits specific to certain data sets.

class starData(object):
    def __init__(self,lightCurve):
        self.lightCurve = lightCurve
        self.analysisResults = dict()