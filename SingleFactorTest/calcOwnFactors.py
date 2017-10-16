#!/Tsan/bin/python
# -*- coding: utf-8 -*-

# Libraries To Use
from __future__ import division
import numpy as np
import pandas as pd
import os
from datetime import datetime, time, date
import config as cf
import h5py

import factorFilterFunctions as ff
data_path = cf.datapath



class CalOwnFactor(object):
    """the parent class for computing new factors,several basic methods are provided in this class"""
    def __init__(self, path):
        """path: STRING ,the path of the data file."""
        self.path = path
        self.datadict = {}

    def addData(self, key, filename):
        """this method is to add DataFrame data into datadict.
        key: STRING ,the key of the data in datadict.
        filename: STRING ,the name of the data file exist in the path,,
        Note that the postfix 'h5' is  necessary here.
        """
        self.datadict[key] = ff.readh5data(self.path, filename)

    def deleteData(self, key):
        """this method is to remove Dataframe data from datadict.
        key: STRING ,the key of the data in datadict.
        """
        try:
            del self.datadict[key]
        except KeyError:
            pass

    def saveData(self, data, newname):
        """this method is to save Dataframe data as h5 file data.
            data: DataFrame ,the data u want to save.
            newname: STRING ,the new save name of the data.
            Note that the postfix 'h5' is not necessary here.
        """
        ff.saveh5data(data, self.path, newname)