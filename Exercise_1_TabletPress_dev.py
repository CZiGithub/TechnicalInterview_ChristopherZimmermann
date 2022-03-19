####################################
# Technical Interview - Exercise 1 #
####################################
# 
# Task:
#   Data Engineering
#   
###################
# Import packages #
###################


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
# from IPython import get_ipython
%matplotlib tk
#get_ipython().run_line_magic('matplotlib', 'inline')

# import dataset as pandas dataframe
df = pd.read_csv('raw_data_tablet_press.csv', sep=',')
# reduce dataset (every nth row) for development phase and store *.csv file 
df = df.iloc[::5, :]
df.to_csv('raw_data_tablet_press_short_5.csv')
# use smaller dataset for development
# df = pd.read_csv('raw_data_tablet_press_short.csv', sep=',')

# print some useful information about the dataset
print(df.shape)
print(df.head(10))
print(df.dtypes)
print(df.nunique())
print(df.isna().sum())

# store data types
str_dtypes = df.dtypes

# format datatypes
df['Unnamed: 10'] = pd.to_datetime((df['Unnamed: 10']), format="%Y-%m-%dT%H:%M:%S.%fZ")
# rename datasets
df = df.rename(columns={'Unnamed: 0': 'id', 'Unnamed: 10': 'time_stamp', 'time': 'variant_type', 'type': 'true_value'})
# sort data
df = df.sort_values(by='time_stamp', ascending=True)


# split dataset for sensor-specific investigation
df_dpMachineStatusS88StatusValue = df[df['sapparametername'].str.contains('dpMachineStatusS88StatusValue') == True]
df_dpMachineStatusS88StatusName = df[df['sapparametername'].str.contains('dpMachineStatusS88StatusName') == True]
# Tablets per hour * 1000:
df_dpMeasuredValueStation1p001 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p001') == True]

df_dpMeasuredValueStation1p002 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p002') == True]
df_dpMeasuredValueStation1p003 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p003') == True]
df_dpMeasuredValueStation1p004 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p004') == True]
df_dpMeasuredValueStation1p005 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p005') == True]
df_dpMeasuredValueStation1p006 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p006') == True]

df_dpMeasuredValueStation1p008 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p008') == True]

df_dpMeasuredValueStation1p013 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p013') == True]

df_dpMeasuredValueStation1p014 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p014') == True]
df_dpMeasuredValueStation1p015 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p015') == True]


# Füllkurve
df_dpMeasuredValueStation1p039 = df[df['sapparametername'].str.contains('dpMeasuredValueStation1p039') == True]


# omit faulty data "None" vs replacing
# Whats the reason for "None"? Should it be ZERO or is it just an error?
df_dpMeasuredValueStation1p001None = df_dpMeasuredValueStation1p001[df_dpMeasuredValueStation1p001['true_value'].str.contains('None') == True]
df_dpMeasuredValueStation1p001NotNone = df_dpMeasuredValueStation1p001[df_dpMeasuredValueStation1p001['true_value'].str.contains('None') == False]
df_dpMeasuredValueStation1p001 = df_dpMeasuredValueStation1p001NotNone

df_dpMeasuredValueStation1p014NotNone = df_dpMeasuredValueStation1p014[df_dpMeasuredValueStation1p014['true_value'].str.contains('None') == False]
df_dpMeasuredValueStation1p014 = df_dpMeasuredValueStation1p014NotNone

df_dpMeasuredValueStation1p015NotNone = df_dpMeasuredValueStation1p015[df_dpMeasuredValueStation1p015['true_value'].str.contains('None') == False]
df_dpMeasuredValueStation1p015 = df_dpMeasuredValueStation1p015NotNone

df_dpMachineStatusS88StatusValueNotNone = df_dpMachineStatusS88StatusValue[df_dpMachineStatusS88StatusValue['true_value'].str.contains('None') == False]
df_dpMachineStatusS88StatusValue = df_dpMachineStatusS88StatusValueNotNone

df_dpMachineStatusS88StatusNameNotNone = df_dpMachineStatusS88StatusName[df_dpMachineStatusS88StatusName['true_value'].str.contains('None') == False]
df_dpMachineStatusS88StatusName = df_dpMachineStatusS88StatusNameNotNone

# replace 'None' by 0
# print(df_dpMeasuredValueStation1p001[df_dpMeasuredValueStation1p001['true_value'].str.contains('None') == True])
# print(df_dpMeasuredValueStation1p001[df_dpMeasuredValueStation1p001['true_value'].str.contains('None') == True].true_value.values)
# # change 'None' to 0 Value?
# mask = (df_dpMeasuredValueStation1p001['true_value'] == 'None')    
# df_dpMeasuredValueStation1p001['true_value'] = df_dpMeasuredValueStation1p001['true_value'].mask(mask,0)
# mask = (df_dpMachineStatusS88StatusValue['true_value'] == 'None')    
# df_dpMachineStatusS88StatusValue['true_value'] = df_dpMachineStatusS88StatusValue['true_value'].mask(mask,0)
# print(df_dpMeasuredValueStation1p001.true_value.values)


####################
# Plotting section # 
####################
# visualize datasets for first analysis

# choose dataset for plotting
x = df_dpMeasuredValueStation1p001.time_stamp.values
x2 = df_dpMachineStatusS88StatusValue.time_stamp.values
x3 = df_dpMeasuredValueStation1p014.time_stamp.values 
#x = df_dpMeasuredValueStation1p001.id.values
y = df_dpMeasuredValueStation1p001.true_value.values
y2 = df_dpMachineStatusS88StatusValue.true_value.values
y3 = df_dpMachineStatusS88StatusName.true_value.values
# print(y2)
# print(y3)
y = df_dpMeasuredValueStation1p001.true_value.astype(float)
y2 = df_dpMachineStatusS88StatusValue.true_value.astype(float)
y3 = df_dpMeasuredValueStation1p014.true_value.astype(float)


############
# Figure 1 #
############
# print(df_dpMachineStatusS88StatusName.true_value)
# fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12,8), gridspec_kw={'height_ratios':[4,2,1]})
plt.figure(1)
#plt.bar(x=df_dpMeasuredValueStation1p001['time_stamp'], height=df_dpMeasuredValueStation1p001['true_value'], color='#087E8B')
# plt.plot(x,y, lw=3, color='#087E8B')
plt.bar(x, y, width=0.025, label='X', color='#087E8B')
plt.title('Data Analysis', size=20)
plt.xticks(rotation=45)
plt.ylim(0,175)
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

par1y = plt.twinx()
par1y.plot(x2, y2, 'k^', markersize = 15, label='S88')
par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
par1y.legend()
par1y.grid(True)

plt.show()


############
# Figure 2 #
############
plt.figure(2)
#plt.bar(x=df_dpMeasuredValueStation1p001['time_stamp'], height=df_dpMeasuredValueStation1p001['true_value'], color='#087E8B')
plt.plot(x,y, lw=3, color='#087E8B')
# plt.bar(x, y, width=0.025, label='X', color='#087E8B')
plt.title('Data Analysis', size=20)
plt.xticks(rotation=45)
# plt.ylim(0,175)
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

par1y = plt.twinx()
par1y.plot(x2, y2, 'k^', markersize = 15, label='S88')
par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
par1y.legend()
par1y.grid(True)

plt.show()

