# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 00:49:56 2019

@author: Mauricio
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

data_frisk = pd.read_excel(r'C:\Users\Mauricio\Documents\Tesis\Algo Bias\Arts\Bias\Frisk\2018_sqf_database.xlsx', sheet_name = r'CY 2018 SQF Web')
data_frisk.groupby(['SUSPECT_RACE_DESCRIPTION']).size()