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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %matplotlib qt
%matplotlib tk
#get_ipython().run_line_magic('matplotlib', 'inline')

today = datetime.datetime.now()
print(today)
# import dataset as pandas dataframe

df = pd.read_csv('PackagingLineDataSet.csv', sep=',', encoding = 'latin1') 
today = datetime.datetime.now()
print(today)

df = df.iloc[::2, :]

# store data types
str_dtypes = df.dtypes


##############
# Clean data #
##############
# format data type from datetime to integer
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScheduledDownTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScheduledDownTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_RunTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_RunTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScrapTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_ScrapTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_QualityTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_QualityTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_OperatingTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_OperatingTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_UnplannedDownTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_UnplannedDownTime_V).dt.total_seconds().astype(int)
df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_LossTime_V = pd.to_timedelta(df.GRZ_L09_04_Cartoner_SX_V0803_01_Efficiency_VC_MZ_803_Z_R_LossTime_V).dt.total_seconds().astype(int)

df = df.sort_values(by='Date_Time', ascending=True)

###########################################
# Method 1 - first target: 'ProductionYN' #
###########################################
#
# prepare datasets
df_X1 = df.drop('ProductionYN', axis=1)
df_X1 = df_X1.drop('Date_Time', axis=1)
df_X1 = df_X1.drop('Category', axis=1)
df_X1 = df_X1.drop('Generated Date', axis=1)
df_X1 = df_X1.drop('ActivityName', axis=1)
df_X1 = df_X1.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty data
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty data


# print(X.head(5))
today = datetime.datetime.now()
print(today)
str_dtypes_df_X1 = df_X1.dtypes


df_y1 = df['ProductionYN']
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25, random_state=4)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
importances1 = pd.DataFrame(data={'Attribute': X_train.columns,'Importance': model.coef_[0]})
importances1 = importances1.sort_values(by='Importance', ascending=False)

plt.figure(1)
plt.bar(x=importances1['Attribute'], height=importances1['Importance'], color='#087E8B')
plt.title('Feature importances', size=20)
plt.xticks(rotation='vertical')
plt.show()

today = datetime.datetime.now()
print(today)
########################################
# Method 1 - second target: 'Category' #
########################################
#
# prepare datasets
df_X2 = df.drop('Category', axis=1)
df_X2 = df_X2.drop('Date_Time', axis=1)
df_X2 = df_X2.drop('Generated Date', axis=1)
df_X2 = df_X2.drop('ActivityName', axis=1)
df_X2 = df_X2.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
df_X2 = df_X2.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
df_X2 = df_X2.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty data
df_X2 = df_X2.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty data

today = datetime.datetime.now()
print(today)
str_dtypes_df_X2 = df_X2.dtypes


df_y2 = df['Category']
X_train, X_test, y_train, y_test = train_test_split(df_X2, df_y2, test_size=0.25, random_state=4)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
importances2 = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model.coef_[0]})
importances2 = importances2.sort_values(by='Importance', ascending=False)

plt.figure(2)
plt.bar(x=importances2['Attribute'], height=importances2['Importance'], color='#087E8B')
plt.title('Feature importances', size=20)
plt.xticks(rotation='vertical')
plt.show()

today = datetime.datetime.now()
print(today)
