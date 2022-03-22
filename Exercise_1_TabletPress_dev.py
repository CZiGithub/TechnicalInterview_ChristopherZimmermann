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
# flag for performing initial clean-up of raw data (flag_cu = 1: clean-up; 
#   flag_cu = 0 load already cleansed dataset)
flag_cu = 1
# plot entire series of datasets (flag_pl_series = 1: perform)
flag_pl_series = 0 # be careful, many plots will be created
# for single plot analysis: enter tagname
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
    df = df.rename(columns={'Unnamed: 0': 'id', 'Unnamed: 10': 'time_stamp',\
                            'time': 'variant_type', 'type': 'true_value'})
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
    df = df.set_index('id')
    # store cleansed dataset
    df.to_csv('raw_data_tablet_press_cleaned.csv')
elif flag_cu == 0:
    df = pd.read_csv('raw_data_tablet_press_cleaned.csv', sep=',')
# convert time stamp
df['time_stamp'] = pd.to_datetime((df['time_stamp']),\
                                  format="%Y-%m-%dT%H:%M:%S.%fZ")
# sort data by time stamp
df = df.sort_values(by='time_stamp', ascending=True)
# drop all rows with "nan"
df = df.dropna()  
# Delete datasets, which have value type of VariantType.Null and store 
#   in separate dataframe
df_null = df[df['variant_type'].str.contains('VariantType.Null') == True]
df = df[df['variant_type'].str.contains('VariantType.Null') == False]
# Transform datatypes labeled with "double" and "int16" to float
df[df['variant_type'].str.contains('VariantType.Double') == True] = \
    df[df['variant_type'].str.contains('VariantType.Double') == \
       True].astype({'true_value': float})
df[df['variant_type'].str.contains('VariantType.Int16') == True] = \
    df[df['variant_type'].str.contains('VariantType.Int16') == \
       True].astype({'true_value': float})

# print some useful information about the dataframe
print(df.shape)
print(df.head(10))
print(df.dtypes)
print(df.nunique())
print(df.isna().sum())

############################################################################
# import tags corresponding to dataset as pandas dataframe (Note: The data #
#    containing the TT_Table needs to be stored manually)                  #
############################################################################
df_TT = pd.read_csv('TT_Table_csv.csv', sep=',')
# create a dictionary with each specific tag/dataset separated
df_dict = {}
# df_TT_dict = {}
for state in df['sapparametername'].unique():
    df_dict[state] = df[df['sapparametername']==state]
# create list of df_dict
list_df_dict = list(df_dict)

####################
# Plotting section # 
####################
# Batch name (unique)
df_dict['dpDataStringBatch'] = df_dict['dpDataStringBatch'].dropna()  

df_dict['dpDataStringBatch'] = \
    df_dict['dpDataStringBatch'].drop_duplicates(subset=['true_value'])
df_dict['dpDiagnoseDiagnosisText'] = \
    df_dict['dpDiagnoseDiagnosisText'].drop_duplicates(subset=['time_stamp'])
df_dict['dpMachineStatusS88StatusName'] = \
    df_dict['dpMachineStatusS88StatusName'].drop_duplicates(subset=['time_stamp'])

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
    + list_df_dict_sorted[43:47] + list_df_dict_sorted[48:63] + \
        list_df_dict_sorted[69:79] + list_df_dict_sorted[80:83] + \
            list_df_dict_sorted[85:88]

# list that will be iteratively plotted
list_df_dict_plot = list_df_dict_nonzero
# list_df_dict_plot = list_df_dict_nonzero[0:10]
# list_df_dict_plot = list_df_dict_sorted

#######################################################
# Automized plotting of each dataset in single figure #
#######################################################
if flag_pl_series == 1:
    for sens_id in list_df_dict_plot:
        # print(sens_id)
        # plot only if datatype is double or int
        if df_dict[sens_id].variant_type.values[0] == 'VariantType.Double'\
            or df_dict[sens_id].variant_type.values[0] == 'VariantType.Int16':
            # extract tagdescription and unit specification of datasets
            get_tagdescription = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
            get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
            get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True]
            get_param_unit = get_param_unit['[EN] Parameter Unit'].values[0]
            # extract data for plotting
            x = df_dict[sens_id].time_stamp.values
            y = df_dict[sens_id].true_value.values.astype(float)
                        
            plt.figure(sens_id)
            
            plt.plot(x,y, '.', markersize = 15, color='#087E8B', label = (get_tagdescription))
            # plt.plot(x,y, '.', markersize = 15, color='#087E8B')
            # plot Batch name:
            # plt.plot()
            x2 = df_dict['dpDataStringBatch'].time_stamp.values
            y2 = df_dict['dpDataStringBatch'].usecaseid.values
            
            x3 = df_dict['dpMachineStatusS88StatusName'].time_stamp.values
            y3 = df_dict['dpMachineStatusS88StatusName'].true_value.values
            # y2 = df_dict['dpDataStringBatch'].usecaseid.values
            # x2 = x2.values
            # plt.plot(x2,y2, 'r*', markersize = 15)
            flg1 = 0
            for idts in x2:
                if flg1 == 0:
                    plt.axvline(idts, color='red', linestyle='--', lw=2.5, label = 'New Batch')
                    # plt.annotate('new batch', xy=(idts,max(y)))
                    flg1 = 1
                else:
                    plt.axvline(idts, color='red', linestyle='--', lw=2.5)
            
                
            # plt.axvline(x2, color='red', linestyle='--')
            plt.title(sens_id, size=20)
            plt.xticks(rotation=45, fontsize = 16)
            plt.xlabel('Time', fontsize=20)
            # plt.ylim(0,105)
            plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']', fontsize=20)
            plt.yticks(fontsize = 16)
            # plt.legend()
            plt.grid(True)
            # ax1.set_xticklabels(x[::2], rotation = 45)
            
            # par1y = plt.twinx()
            # par1y.plot(x2, y2, 'k*', markersize = 10, label='S88')
            # par1y.set_ylabel('Idle/Complete - Running - Held - Paused')
            # # par1y.legend()
            # par1y.grid(True)
            plt.show()          
            # plt.tight_layout() 
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            sens_id_clean = sens_id.replace('/','_')
            # plt.savefig('./Results/' + (get_tagdescription_clean) + ".png", dpi = 100)
            # plt.savefig('./Results/' + (get_tagdescription_clean) + ".pdf", dpi = 100)
            # plt.savefig('./Results/' + (sens_id_clean) + ".jpeg", dpi = 100)
            # plt.close("all")
            

# plt.close("all")

###########################
# plot individual figures # 
###########################

sensor_analysis = 'dpMeasuredValueStation1p014'
sensor_analysis2 = 'dpMeasuredValueStation1p015'

sens_id = sensor_analysis
sens_id2 = sensor_analysis2
get_tagdescription = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True].tagdescription.values[0]
get_param_unit = df_TT[df_TT['sapparametername'].str.contains(sens_id) == True]
get_param_unit = get_param_unit['[EN] Parameter Unit'].values[0]

get_tagdescription2 = df_TT[df_TT['sapparametername'].str.contains(sens_id2) == True].tagdescription.values[0]
get_param_unit2 = df_TT[df_TT['sapparametername'].str.contains(sens_id2) == True].tagdescription.values[0]
get_param_unit2 = df_TT[df_TT['sapparametername'].str.contains(sens_id2) == True]
get_param_unit2 = get_param_unit2['[EN] Parameter Unit'].values[0]
# extract data for plotting
x = df_dict[sens_id].time_stamp.values
y = df_dict[sens_id].true_value.values.astype(float)

xB = df_dict[sens_id2].time_stamp.values
yB = df_dict[sens_id2].true_value.values.astype(float)

plt.figure(sens_id)

plt.plot(x,y, '.', markersize = 15, color='#087E8B', label = (get_tagdescription))
# plt.plot(x,y, '.', markersize = 15, color='#087E8B')
# plot Batch name:
# plt.plot()
x2 = df_dict['dpDataStringBatch'].time_stamp.values
y2 = df_dict['dpDataStringBatch'].usecaseid.values

x3 = df_dict['dpMachineStatusS88StatusName'].time_stamp.values
y3 = df_dict['dpMachineStatusS88StatusName'].true_value.values
# y2 = df_dict['dpDataStringBatch'].usecaseid.values
# x2 = x2.values
# plt.plot(x2,y2, 'r*', markersize = 15)
flg1 = 0
for idts in x2:
    if flg1 == 0:
        plt.axvline(idts, color='red', linestyle='--', lw=2.5, label = 'New Batch')
        # plt.annotate('new batch', xy=(idts,max(y)))
        flg1 = 1
    else:
        plt.axvline(idts, color='red', linestyle='--', lw=2.5)

flg1 = 0
id_it = 0
for idts in x3:
    # if y3[id_it] == 'Held'
    # plt.annotate(y3[id_it], xy=(idts,max(y)))
    id_it +=1
    
    #     plt.axvline(idts, linestyle='--', lw=2.5)
# plt.axvline(x2, color='red', linestyle='--')
plt.title(sens_id, size=20)
plt.xticks(rotation=45, fontsize = 16)
plt.xlabel('Time', fontsize=20)
# plt.ylim(0,105)
plt.ylabel(get_tagdescription + ' [' + get_param_unit + ']', fontsize=20)
plt.yticks(fontsize = 16)
# plt.legend()
plt.grid(True)
# ax1.set_xticklabels(x[::2], rotation = 45)

# par1y = plt.twinx()
# par1y.plot(xB, yB, '*', color = 'darkblue', markersize = 3, label='S88')
# par1y.set_ylabel(get_tagdescription2 + ' [' + get_param_unit2 + ']', fontsize=20)
# # par1y.set_yticks(size = 16)
# # # par1y.legend()
# par1y.grid(True)
plt.show()          
plt.tight_layout() 
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
sens_id_clean = sens_id.replace('/','_')
# plt.savefig('./Results/' + (get_tagdescription_clean) + ".png", dpi = 100)
# plt.savefig('./Results/' + (get_tagdescription_clean) + ".pdf", dpi = 100)
# plt.savefig('./Results/' + (sens_id_clean) + "_2z1.jpeg", dpi = 100)

# plt.close("all")