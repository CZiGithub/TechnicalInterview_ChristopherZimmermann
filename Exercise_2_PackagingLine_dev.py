####################################
# Technical Interview - Exercise 2 #
####################################
# 
# Task:
#   Feature importance and selection
#   
###################
# Import packages #
###################

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from IPython import get_ipython
# %matplotlib qt
%matplotlib tk

# import dataset as pandas dataframe
# df = pd.read_csv('PackagingLineDataSet.csv', encoding = 'latin1')
# reduce dataset (every nth row) for development phase and store *.csv file 
# df = df.iloc[::5, :]
# df.to_csv('PackagingLineDataSet_short_20.csv')
# use smaller dataset for development
df = pd.read_csv('PackagingLineDataSet_short_10.csv', sep=',', encoding = 'latin1')

# print some useful information about the dataset
print(df.shape)
print(df.head(10))
print(df.dtypes)
print(df.nunique()) 
print(df.isna().sum())
 
# store data types
str_dtypes = df.dtypes

# split dataset to calculate feature importance:
X = df.drop('ProductionYN', axis=1) # target 1
X = X.drop('Category', axis=1) # target 2
# first step: drop some features that not directly usable datatypes
#   next: transform datatype! further investigation needed
X = X.drop('Date_Time', axis=1) 
X = X.drop('Generated Date', axis=1)
X = X.drop('ActivityName', axis=1)
X = X.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScheduledDownTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_RunTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScrapTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_QualityTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_OperatingTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_UnplannedDownTime_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_LossTime_V', axis=1)
X = X.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1)
X = X.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1)

y1 = df['ProductionYN']
y2 = df['Category']

print(X.head(5))
print(y1.head(5))
print(y2.head(5))

str_dtypes_X = X.dtypes
str_dtypes_y1 = y1.dtypes
str_dtypes_y2 = y2.dtypes

