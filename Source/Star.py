# This is a star object. It contains a dict that stores different data
# sets taken observing the star. The data sets are formated as a starData
# object.
#
# The name variable is just to allow a marker, but is not necissary.
# The position is sometimes fed into a function that fetches a data
# set.

class Star(object):
    def __init__(self,name,position):
        self.name = name
        self.position = position
        self.dataSets = dict()