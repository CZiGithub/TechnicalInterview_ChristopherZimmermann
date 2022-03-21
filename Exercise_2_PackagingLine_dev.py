####################################
# Technical Interview - Exercise 2 #
####################################
# 
# Task:
#   1.) Feature importance and selection
#   2.) Predictive model development
#   to do: code clean-up, some sections are redundant
#   to do: code documentation
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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
# %matplotlib qt
%matplotlib tk
#get_ipython().run_line_magic('matplotlib', 'inline')

today = datetime.datetime.now()
print(today)
# import dataset as pandas dataframe
df = pd.read_csv('PackagingLineDataSet.csv', sep=',', encoding = 'latin1') 
today = datetime.datetime.now()
print(today)
# for development phase: use only a smaller chunk of the data
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

########################
# Validating the model #
########################
# make predictions
y_pred=model_LG_Prod.predict(X_test_scaled)
# Accuracy
print("LG - first target: 'ProductionYN' - Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Recall
print("LG - first target: 'ProductionYN' - Recall:",metrics.recall_score(y_test, y_pred))
# Precision
print("LG - first target: 'ProductionYN' - Precision:",metrics.precision_score(y_test, y_pred))
# F1
print("LG - first target: 'ProductionYN' - Recall:",metrics.f1_score(y_test, y_pred))

# cross-validation score
scores = cross_val_score(model_LG_Prod, X_train_scaled, y_train, cv=2)
print("LG - first target: 'ProductionYN' - Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=2, shuffle=True)
kf_cv_scores = cross_val_score(model_LG_Prod, X_train_scaled, y_train, cv=kfold )
print("LG - first target: 'ProductionYN' - K-fold CV average score: %.2f" % kf_cv_scores.mean())



########################################
# Method 1 - second target: 'Category' #
########################################
#
# prepare datasets
df_X1 = df.drop('Category', axis=1)
df_X1 = df_X1.drop('ProductionYN', axis=1)
df_X1 = df_X1.drop('Date_Time', axis=1)
df_X1 = df_X1.drop('Generated Date', axis=1)
df_X1 = df_X1.drop('ActivityName', axis=1)
df_X1 = df_X1.drop('GRZ_L09_05_Seidenader_Recipe_Recipe__1__V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_02_Filler_APP_V0931_00_BatchProtocol_VC_MZ_931_EEZ_S_BatchName_V', axis=1)
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacControlLift_V', axis=1) # faulty data
df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_VC_MZ_456_A_VacMonitor_V', axis=1) # faulty data

# define second target
df_y1 = df['Category']
# split dataset
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25, random_state=4)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

# another option would be one-versus-rest, if focus on "2 unplanned downtime" 
model_LG_Cat = LogisticRegression(multi_class = 'multinomial', max_iter=2000)
model_LG_Cat.fit(X_train_scaled, y_train)

# make predictions
y_pred=model_LG_Cat.predict(X_test_scaled)
# importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[0]})
# importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[1]})
# importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[2]})
# importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[3]})
# importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[4]})
importances_LG_Cat = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model_LG_Cat.coef_[0]}) # "2 unplanned downtime"
importances_LG_Cat = importances_LG_Cat.sort_values(by='Importance', ascending=False)

# save importances
importances_LG_Cat.to_csv('importances_LG_Cat.csv')

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

########################
# Validating the model #
########################
# make predictions
y_pred=model_LG_Cat.predict(X_test_scaled)
# Accuracy
print("LG - second target: 'Category' - Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Recall
print("LG - second target: 'Category' - Recall:",metrics.recall_score(y_test, y_pred,average = None))
# Precision
print("LG - second target: 'Category' - Precision:",metrics.precision_score(y_test, y_pred,average = None))
# F1
print("LG - second target: 'Category' - Recall:",metrics.f1_score(y_test, y_pred,average = None))

# cross-validation score
scores = cross_val_score(model_LG_Cat, X_train_scaled, y_train, cv=2)
print("LG - second target: 'Category' - Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=2, shuffle=True)
kf_cv_scores = cross_val_score(model_LG_Cat, X_train_scaled, y_train, cv=kfold )
print("LG - second target: 'Category' - K-fold CV average score: %.2f" % kf_cv_scores.mean())

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

########################
# Validating the model #
########################
# make predictions
y_pred=model_LG_Prod.predict(X_test_scaled)
# Accuracy
print("XGB - first target: 'ProductionYN' - Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Recall
print("XGB - first target: 'ProductionYN' - Recall:",metrics.recall_score(y_test, y_pred))
# Precision
print("XGB - first target: 'ProductionYN' - Precision:",metrics.precision_score(y_test, y_pred))
# F1
print("XGB - first target: 'ProductionYN' - Recall:",metrics.f1_score(y_test, y_pred))

# cross-validation score
scores = cross_val_score(model_XGB_Prod, X_train_scaled, y_train, cv=2)
print("XGB - first target: 'ProductionYN' - Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=2, shuffle=True)
kf_cv_scores = cross_val_score(model_XGB_Prod, X_train_scaled, y_train, cv=kfold )
print("XGB - first target: 'ProductionYN' - K-fold CV average score: %.2f" % kf_cv_scores.mean())

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

model_XGB_Cat = XGBClassifier(objective='multi:softprob')
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

########################
# Validating the model #
########################
# make predictions
y_pred=model_XGB_Cat.predict(X_test_scaled)
# Accuracy
print("XGB - second target: 'Category' - Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Recall
print("XGB - second target: 'Category' - Recall:",metrics.recall_score(y_test, y_pred,average = None))
# Precision
print("XGB - second target: 'Category' - Precision:",metrics.precision_score(y_test, y_pred,average = None))
# F1
print("XGB - second target: 'Category' - Recall:",metrics.f1_score(y_test, y_pred,average = None))

# cross-validation score
scores = cross_val_score(model_XGB_Cat, X_train_scaled, y_train, cv=2)
print("XGB - second target: 'Category' - Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=2, shuffle=True)
kf_cv_scores = cross_val_score(model_XGB_Cat, X_train_scaled, y_train, cv=kfold )
print("XGB - second target: 'Category' - K-fold CV average score: %.2f" % kf_cv_scores.mean())


####################
# Predictive Model #
####################
# KNeighborsClassifier
# tbd....

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
#################################
# now drop unimportant features #
#################################
# 
# unimportant_features = importances_XGB_Cat[abs(importances_XGB_Cat['Importance'].values) <= 0.025*max(abs(importances_XGB_Cat['Importance'].values))]
unimportant_features = importances_LG_Cat[abs(importances_XGB_Cat['Importance'].values) <= 0.025*max(abs(importances_XGB_Cat['Importance'].values))]
df_X1 = df_X1.drop(unimportant_features.Attribute, axis=1)


# define first target
df_y1 = df['Category']
# df_y1 = df['GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_IX_456_B5VacLow_V']
# df_X1 = df_X1.drop('GRZ_L09_04_Cartoner_SX_V0456_01_RobotTool_IX_456_B5VacLow_V', axis=1)

#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25,random_state=4)

# scale
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_KNN=KNeighborsClassifier(n_neighbors=7, metric='euclidean')
model_KNN.fit(X_train_scaled,y_train)

# make predictions
y_pred=model_KNN.predict(X_test_scaled)

########################
# Validating the model #
########################

# Accuracy
print("KNNClassifier - second target: 'Category' - Accuracy:",model_KNN.score(X_test_scaled, y_test))
# Accuracy
print("KNNClassifier - second target: 'Category' - Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Recall
print("KNNClassifier - second target: 'Category' - Recall:",metrics.recall_score(y_test, y_pred,average = None))
# Precision
print("KNNClassifier - second target: 'Category' - Precision:",metrics.precision_score(y_test, y_pred,average = None))
# F1
print("KNNClassifier - second target: 'Category' - F1:",metrics.f1_score(y_test, y_pred,average = None))

# cross-validation score
scores = cross_val_score(model_KNN, X_train_scaled, y_train, cv=2)
print("KNNClassifier - second target: 'Category' - Mean cross-validation score: %.2f" % scores.mean())
kfold = KFold(n_splits=2, shuffle=True)
kf_cv_scores = cross_val_score(model_KNN, X_train_scaled, y_train, cv=kfold )
print("KNNClassifier - second target: 'Category' - K-fold CV average score: %.2f" % kf_cv_scores.mean())



####################
# Predictive Model #
####################
# KNeighborsRegressor
# tbd...

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
# df_y1 = df['Category']
df_y1 = df['GRZ_L09_01_Blistering_PLC_Modules__Default__Global_PV_tPeCupEdSW_fLatchDiffMm_V']
df_X1 = df_X1.drop('GRZ_L09_01_Blistering_PLC_Modules__Default__Global_PV_tPeCupEdSW_fLatchDiffMm_V', axis=1)

#Split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(df_X1, df_y1, test_size=0.25,random_state=4)

# scale
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model_KNN_reg=KNeighborsRegressor(n_neighbors=7, metric='euclidean')
model_KNN_reg.fit(X_train_scaled,y_train)

# make predictions
y_pred=model_KNN_reg.predict(X_test_scaled)

########################
# Validating the model #
########################
# Accuracy
print("KNNRegressor - continuous target: 'Blistering' - Accuracy:",model_KNN_reg.score(X_test_scaled, y_test))

