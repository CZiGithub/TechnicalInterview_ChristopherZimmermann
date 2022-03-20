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
from xgboost import XGBClassifier
# %matplotlib qt
%matplotlib tk
#get_ipython().run_line_magic('matplotlib', 'inline')

today = datetime.datetime.now()
print(today)
# import dataset as pandas dataframe
df = pd.read_csv('PackagingLineDataSet.csv', sep=',', encoding = 'latin1') 
today = datetime.datetime.now()
print(today)
# for development phase: only a smaller chunk of the data
df = df.iloc[::200, :]



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
df.Date_Time = pd.to_datetime(df.Date_Time).astype(int) # maybe further scaling

df = df.sort_values(by='Date_Time', ascending=True) # might not be necessary

# replace datasets and change datatype from string to int
df['Category'] = df['Category'].replace('2 Unplanned downtime',2)
df['Category'] = df['Category'].replace('1 Planned downtime',1)
df['Category'] = df['Category'].replace('3 Changeovers',3)
df['Category'] = df['Category'].replace('4 Production time',4)
df['Category'] = df['Category'].replace('5 Microstop',5)

# store data types
str_dtypes = df.dtypes


##################################
# Feature importance calculation #
##################################
# Methods:
#   1.) Logistic regression
#   2.) XGBClassifier
#
# Targets:
#   1.) ProductionYN
#   2.) Category
#
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
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty dataset
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty dataset

# define first target
df_y1 = df['ProductionYN']
# split dataset
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25, random_state=4)

# scale
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_LG_Prod = LogisticRegression(max_iter=2000)
model_LG_Prod.fit(X_train_scaled, y_train)
importances_LG_Prod = pd.DataFrame(data={'Attribute': X_train.columns,'Importance': model_LG_Prod.coef_[0]})
importances_LG_Prod = importances_LG_Prod.sort_values(by='Importance', ascending=False)

# save importances
importances_LG_Prod.to_csv('importances_LG_ProductionYN.csv')

plt.figure(1)
plt.bar(x=importances_LG_Prod['Attribute'], height=importances_LG_Prod['Importance'], color='#087E8B')
plt.title('Feature importances - LG - ProductionYN', size=20)
# plt.xticks(rotation='vertical')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()

today = datetime.datetime.now()
print(today)
########################################
# Method 1 - second target: 'Category' #
########################################
#
# prepare datasets
df_X2 = df.drop('Category', axis=1)
df_X2 = df_X2.drop('ProductionYN', axis=1)
df_X2 = df_X2.drop('Date_Time', axis=1)
df_X2 = df_X2.drop('Generated Date', axis=1)
df_X2 = df_X2.drop('ActivityName', axis=1)
df_X2 = df_X2.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
df_X2 = df_X2.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
df_X2 = df_X2.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty data
df_X2 = df_X2.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty data

# define second target
df_y2 = df['Category']
# split dataset
X_train, X_test, y_train, y_test = train_test_split(df_X2, df_y2, test_size=0.25, random_state=4)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_LG_Cat = LogisticRegression(max_iter=2000)
model_LG_Cat.fit(X_train_scaled, y_train)
importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[0]})
importances_LG_Cat = importances_LG_Cat.sort_values(by='Importance', ascending=False)

# save importances
importances_LG_Cat.to_csv('importances_LG_Category.csv')

plt.figure(2)
plt.bar(x=importances_LG_Cat['Attribute'], height=importances_LG_Cat['Importance'], color='#087E8B')
plt.title('Feature importances - LG - Category', size=20)
# plt.xticks(rotation='vertical')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()

today = datetime.datetime.now()
print(today)

###########################################
# Method 2 - first target: 'ProductionYN' #
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
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty dataset
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty dataset


# define first target
df_y1 = df['ProductionYN']
# split dataset
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25, random_state=4)

# scale
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_XGB_Prod = XGBClassifier()
model_XGB_Prod.fit(X_train_scaled, y_train)
importances_XGB_Prod = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_XGB_Prod.feature_importances_})
importances_XGB_Prod = importances_XGB_Prod.sort_values(by='Importance', ascending=False)

# save importances
importances_XGB_Prod.to_csv('importances_XGB_ProductionYN.csv')

today = datetime.datetime.now()
print(today)

plt.figure(3)
plt.bar(x=importances_XGB_Prod['Attribute'], height=importances_XGB_Prod['Importance'], color='#087E8B')
plt.title('Feature importances - XGB - ProductionYN', size=20)
# plt.xticks(rotation='vertical')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()


########################################
# Method 2 - second target: 'Category' #
########################################
#
# prepare the datasets
df_X1 = df.drop('ProductionYN', axis=1)
df_X1 = df_X1.drop('Date_Time', axis=1)
df_X1 = df_X1.drop('Category', axis=1)
df_X1 = df_X1.drop('Generated Date', axis=1)
df_X1 = df_X1.drop('ActivityName', axis=1)
df_X1 = df_X1.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty dataset
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty dataset

# define first target
df_y1 = df['Category']
# split dataset
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25, random_state=4)

# scale
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_XGB_Cat = XGBClassifier()
model_XGB_Cat.fit(X_train_scaled, y_train)
importances_XGB_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_XGB_Cat.feature_importances_})
importances_XGB_Cat = importances_XGB_Cat.sort_values(by='Importance', ascending=False)

# save importances
importances_XGB_Cat.to_csv('importances_XGB_Category.csv')

today = datetime.datetime.now()
print(today)

plt.figure(4)
plt.bar(x=importances_XGB_Cat['Attribute'], height=importances_XGB_Cat['Importance'], color='#087E8B')
plt.title('Feature importances - XGB - Category', size=20)
# plt.xticks(rotation='vertical')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
today = datetime.datetime.now()
print(today)

