####################################
# Technical Interview - Exercise 1 #
####################################
# 
# Task:
#   Data Engineering
#   Clean & prepare the raw machine data of the tablet press. 
#
#########################
# Configuration section #
#########################
# flag for performing initial clean-up of raw data (flag_cu = 1: clean-up; flag_cu = 0 load already cleansed dataset)
flag_cu = 0
# plot entire series of datasets (flag_pl_series = 1: perform)
flag_pl_series = 1 # be careful, many plots will be created
# for single plot analysis: enter sensorname
sensor_analysis = 'dpMeasuredValueStation1p001'

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

#########################
# Import and clean data #
#########################
# clean data only if flag is active
if flag_cu == 1:
    # import dataset as pandas dataframe
    df = pd.read_csv('raw_data_tablet_press.csv', sep=',')
    # give columns meaningful names
    df = df.rename(columns={'Unnamed: 0': 'id', 'Unnamed: 10': 'time_stamp', 'time': 'variant_type', 'type': 'true_value'})
    # format datatypes (time stamp)
    df_nan = df[df['variant_type'].isnull() == True]
    # repair dataset
    for repair_line in df_nan['id']: # slow looping but it is doing the job
        # split string into meaningful data
        temp_str = df[df['id'] == repair_line].value.values[0]
        temp_str_split_1 = temp_str.split(',"')
        temp_str_split_2 = temp_str_split_1[1].split('",')
        temp_str_split_3 = temp_str_split_2[1].split(',')
        
        # insert correct values in dataframe
        df.loc[repair_line,'value'] = temp_str_split_1[0]
        df.loc[repair_line,'true_value'] = temp_str_split_2[0]
        df.loc[repair_line,'variant_type'] = temp_str_split_3[0]
        df.loc[repair_line,'time_stamp'] = temp_str_split_3[1]
    # store cleansed dataset
    df.to_csv('raw_data_tablet_press_cleaned.csv')
elif flag_cu == 0:
    df = pd.read_csv('raw_data_tablet_press_cleaned.csv', sep=',')

# convert time stamp
df['time_stamp'] = pd.to_datetime((df['time_stamp']), format="%Y-%m-%dT%H:%M:%S.%fZ")
# sort data by time stamp
df = df.sort_values(by='time_stamp', ascending=True)
# Delete datasets, which have value type of VariantType.Null and and store in separate frame
df_null = df[df['variant_type'].str.contains('VariantType.Null') == True]
df = df[df['variant_type'].str.contains('VariantType.Null') == False]
# Transform datatypes labeled with "double" and "int16" to float
df[df['variant_type'].str.contains('VariantType.Double') == True] = df[df['variant_type'].str.contains('VariantType.Double') == True].astype({'true_value': float})
df[df['variant_type'].str.contains('VariantType.Int16') == True] = df[df['variant_type'].str.contains('VariantType.Int16') == True].astype({'true_value': float})

# reduce dataset (every nth row) for development phase
# df = df.iloc[::10, :]
# use smaller dataset for development
# df = pd.read_csv('raw_data_tablet_press_short.csv')
#print(df)
# import tags corresponding to dataset as pandas dataframe
df_TT = pd.read_csv('TT_Table_csv.csv', sep=',')

# print some useful information about the dataset
print(df.shape)
print(df.head(10))
print(df.dtypes)
print(df.nunique())
print(df.isna().sum())

# store existing data types for further investigation
str_dtypes = df.dtypes

# create a dictionary with each specific dataset separated
df_dict = {}
# df_TT_dict = {}
for state in df['sapparametername'].unique():
    df_dict[state] = df[df['sapparametername']==state]
# create list of df_dict
list_df_dict = list(df_dict)

####################
# Plotting section # 
####################

########################################################
# Visualize all individual datasets for first analysis # 
########################################################
# create auxiliary lists of datasets
list_df_dict_sorted = sorted(list_df_dict)
# list containing only measured data
list_df_dict_measured = list_df_dict_sorted[13:46]
# list containing only set data
list_df_dict_set = list_df_dict_sorted[47:99]
# list containing all data that have at least one nonzero value
list_df_dict_nonzero = list_df_dict_sorted[0:29] + list_df_dict_sorted[38:41] \
    + list_df_dict_sorted[43:47] + list_df_dict_sorted[48:63] + list_df_dict_sorted[69:79] \
        + list_df_dict_sorted[80:83] + list_df_dict_sorted[85:88]

# list that will be iteratively plotted
list_df_dict_plot = list_df_dict_nonzero
# list_df_dict_plot = list_df_dict_nonzero[0:10]
list_df_dict_plot = list_df_dict_sorted

if flag_pl_series == 1:
    for sens_id in list_df_dict_plot:
        # print(sens_id)
        # plot only if datatype is double or int
        if df_dict[sens_id].variant_type.values[0] == 'VariantType.Double' or df_dict[sens_id].variant_type.values[0] == 'VariantType.Int16':
            # extract tagdescription and unit specification of datasets
            get_tagdescription = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
            get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
            get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True]
            get_param_unit = get_param_unit['[EN] Parameter Unit'].values[0]
            # extract data for plotting
            x = df_dict[sens_id].time_stamp.values
            y = df_dict[sens_id].true_value.values.astype(float)
            
            # x = df_dict[sens_id].time_stamp.values
            # y = df_dict[sens_id].true_value.values.astype(float)
            
            plt.figure(sens_id)
            
            plt.plot(x,y, '.', markersize = 10, color='#087E8B', label = (get_tagdescription))
            # plt.plot(x,y, 10, markersize = 10, color='orange', label = (get_tagdescription))
            # plt.bar(x, y, width=0.025, label='X', color='#087E8B')
            plt.title(sens_id, size=20)
            plt.xticks(rotation=45)
            # plt.ylim(0,105)
            plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']')
            # plt.yticks((0,100))
            # plt.legend()
            plt.grid(True)
            # ax1.set_xticklabels(x[::2], rotation = 45)
            
            # par1y = plt.twinx()
            # par1y.plot(x2, y2, 'k*', markersize = 10, label='S88')
            # par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
            # # par1y.legend()
            # par1y.grid(True)
            plt.show()

# plt.close("all")
###################################
# Plot with pandas dataframe.plot #
###################################
# df_dict[sensor_analysis].plot(x ='time_stamp', y='true_value', kind = 'scatter')
# # df_dict[sensor_analysis].plot(x ='time_stamp', y='true_value', kind = 'scatter')
# plt.show()

# extract data for individual plotting
x = df_dict[sensor_analysis].time_stamp.values
y = df_dict[sensor_analysis].true_value.values.astype(float)

x2 = df_dict['dpMachineStatusS88StatusValue'].time_stamp.values
y2 = df_dict['dpMachineStatusS88StatusValue'].true_value.values.astype(float)

# extract tagdescription and unit specification of datasets
get_tagdescription = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sensor_analysis) == True]
get_param_unit = get_param_unit['[EN] Parameter Unit'].values[0]

############
# Figure 1 #
############
plt.figure(2)
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
plt.figure(3)
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
plt.figure(4)
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
