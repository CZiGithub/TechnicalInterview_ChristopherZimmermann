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
# df = df.iloc[::5, :]
# df.to_csv('raw_data_tablet_press_short_5.csv')
# use smaller dataset for development
# df = pd.read_csv('raw_data_tablet_press_short.csv', sep=',')
# import tags corresponding to dataset as pandas dataframe
# note: *.csv file was extracted manually from Tag Translation Table - sheet
df_TT = pd.read_csv('TT_Table_csv.csv', sep=',')

# enter here, which dataset you want to visualize
sensor_analysis = 'dpMeasuredValueStation1p001'

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


##############
# Clean data #
##############
print(df.shape)
# Omit datasets, in which values that are stated as "None"
df = df[df['true_value'].str.contains('None') == False]
# print(df['variant_type'].unique())
# print(df.shape)
# # Delete datasets, which have value type of NAN
# df = df[df['variant_type'].str.contains('nan') == False] # similar result as with the "none" delete
# Transform datatypes labeled with "double" and "int16" to float
# create a list of all data types in data set
# print(df['variant_type'].unique())
list_variant_types = df['variant_type'].unique()
df[df['variant_type'].str.contains('VariantType.Double') == True].astype({'true_value': float})
df[df['variant_type'].str.contains('VariantType.Int16') == True].astype({'true_value': float})
print(df.shape)
# store existing data types for further investigation
str_dtypes = df.dtypes
# extract tagdescription and unit specification of datasets
get_tagdescription = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True]
get_param_unit = get_param_unit['[EN] Parameter Unit'].values[0]

# create a dictionary with each specific dataset separated
df_dict = {}
# df_TT_dict = {}
for state in df['sapparametername'].unique():
    df_dict[state]=df[df['sapparametername']==state]

# create list of df_dict
list_df_dict = list(df_dict)
# print(list_df_dict[0])
# print(df_dict[list_df_dict[0]])

####################
# Plotting section # 
####################
x = df_dict[sensor_analysis].time_stamp.values
y = df_dict[sensor_analysis].true_value.values.astype(float)

x2 = df_dict['dpMachineStatusS88StatusValue'].time_stamp.values
y2 = df_dict['dpMachineStatusS88StatusValue'].true_value.values.astype(float)

############
# Figure 1 #
############
plt.figure(1)
plt.plot(x,y, lw=3, color='#087E8B', label = (get_tagdescription))
# plt.bar(x, y, width=0.025, label='X', color='#087E8B')
plt.title(sensor_analysis, size=20)
plt.xticks(rotation=45)
# plt.ylim(0,105)
plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']')
# plt.yticks((0,100))
plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

par1y = plt.twinx()
par1y.plot(x2, y2, 'k*', markersize = 10, label='S88')
par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
# par1y.legend()
par1y.grid(True)
# par1y.set_ylim(0,4)
plt.show()


############
# Figure 2 #
############
plt.figure(2)
#plt.bar(x=df_dpMeasuredValueStation1p001['time_stamp'], height=df_dpMeasuredValueStation1p001['true_value'], color='#087E8B')
# plt.plot(x,y, lw=3, color='#087E8B')
plt.bar(x, y, width=0.025, color='#087E8B', label = (get_tagdescription))
plt.title(sensor_analysis, size=20)
plt.xticks(rotation=45)
# plt.ylim(0,175)
plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']')
plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

par1y = plt.twinx()
par1y.plot(x2, y2, 'k*', markersize = 10, label='S88')
par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
# par1y.legend()
par1y.grid(True)
plt.show()

############
# Figure 3 #
############
plt.figure(3)
plt.plot(x, y, '+', markersize = 10, color='#087E8B', label = (get_tagdescription))
plt.title(sensor_analysis, size=20)
plt.xticks(rotation=45)
# plt.ylim(0,175)
plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']')
plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

par1y = plt.twinx()
par1y.plot(x2, y2, 'k*', markersize = 10, label='S88')
par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
# par1y.legend()
par1y.grid(True)
plt.show()
