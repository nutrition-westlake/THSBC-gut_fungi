
"""
Created on Sat Mar  4 00:23:44 2023

@author: Ke Zhang
"""


## 确认python的版本运行与终端运行的版本一致【python --version】
import sys; print(sys.version)



## import the needed library 

import pandas as pd
from pandas import ExcelWriter
from pandas import Series, DataFrame

import numpy as np
import os


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn import metrics 
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.metrics import r2_score, precision_recall_curve, explained_variance_score
from sklearn.metrics import mean_squared_error


import lightgbm as lgb
from lightgbm import LGBMClassifier 

import shap
import scipy.stats as stats 
from scipy.stats import spearmanr, pearsonr

shap.initjs()










# Pre-define 
# -----------------------------------------------------------------------------

'''
calculate MAPE, Mean Absolute Percentage Error（平均绝对百分比误差）
'''

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



'''
kfold split samples based on individuals, 按照个人进行样本随机拆分
'''

def id_kfold(data, id, folds, seed): 
    idlist = list(set(id))
    np.random.seed(seed) 
    shuffled_idlsit = np.random.permutation(idlist) # 随机打乱向量  

    kfold = pd.DataFrame(np.array_split(shuffled_idlsit, folds))
    kfold_t = kfold.transpose()    

    kfold_melt = kfold_t.melt()
    kfold_melt = kfold_melt.dropna()
    kfold_melt.value = kfold_melt.value.astype(object)
    kfold_melt.columns = ['kfold', 'id']
    kfold_melt.dtypes

    input_data = pd.merge(kfold_melt, data, on='id', how='left')

    return input_data 




'''
calculate overall mean shap values for features importance ranking
'''

def calculate_meanSHAP(shap_fold, seed): 
    # meanSHAP_all = pd.DataFrame()

    shap_abs = abs(shap_fold)
    shap_mean = pd.DataFrame(shap_abs.mean())
    shap_mean.columns = [seed]
    meanSHAP_all = pd.concat([meanSHAP_all, shap_mean], axis=1)

    return meanSHAP_all 




'''
merge predicted values and corresponding true values
'''

def merge_pred_true(folds, tests, preds, seed): 
    output_test = pd.DataFrame()
    for i in np.arange(0,folds):
        tests[i] = pd.DataFrame(tests[i])
        output_test = pd.concat([output_test, tests[i]])
    
    nrow = output_test.iloc[:,0].size
    output_test.index = np.arange(0,nrow)
    output_test.columns = ['true']


    output_pred = pd.DataFrame()
    for i in np.arange(0,folds):
        preds[i] = pd.DataFrame(preds[i])
        output_pred = pd.concat([output_pred, preds[i]], axis=0)
    
    nrow = output_pred.iloc[:,0].size
    output_pred.index = np.arange(0,nrow)  
    output_pred.columns = ['pred']
    
    output_total = []
    output_total = pd.concat([output_pred, output_test], axis=1)
    output_total.columns = [('pred_' + str(seed) ),('test_' + str(seed))]

    # output_prediction_all = pd.DataFrame()
    output_prediction_all = pd.concat([output_prediction_all, output_total], axis=1)

    return output_prediction_all, output_test, output_pred




'''
evaluate performance of prediction models
'''

def eval_perf(y_true, y_pred, seed): 
    model_eval = pd.DataFrame()

    model_eval.loc[seed, 'Coefficient_of_determination'] = r2_score(y_true=y_true, y_pred=y_pred) 
    model_eval.loc[seed, 'explained_variance_score'] = explained_variance_score(y_true=y_true, y_pred=y_pred)

    model_eval.loc[seed, 'pearson_r'], model_eval.loc[seed, 'pearson_p'] = pearsonr(y_true, y_pred)
    model_eval.loc[seed, 'spearman_r'], model_eval.loc[seed, 'spearman_p'] = spearmanr(y_true, y_pred)
    model_eval.loc[seed, 'RMSE'] = mean_squared_error(y_true=y_true, y_pred=y_pred)
    model_eval.loc[seed, 'MAPE'] = calculate_mape(y_true=y_true, y_pred=y_pred)
    
    # model_eval_all = pd.DataFrame()
    model_eval_all = pd.concat([model_eval_all, model_eval], axis=0)

    return model_eval_all 













    


# Parameters settngs
# -----------------------------------------------------------------------------

root = '/Users/zhangke/Desktop/result_prediction_240313'
os.chdir(root)

# random_seedlist = pd.read_excel('lightGBM_random_seedlist.xlsx') 
seed = 123456
date = '230912'


# set hyperparameters

lgb_params = {
              'objective': 'binary',
              'metric': 'auc',
              'learning_rate': 0.005,
              'feature_fraction': 0.2,
              'min_data_in_leaf': 15, 
              'early_stopping_rounds': None, 'n_estimators': 2000,
              'bagging_fraction': 0.8, 'bagging_freq': 1,
              'num_threads': 1, 'verbose': -1, 'silent': True}


folds = 10
n_bs = 1



data = pd.read_stata('ITS_revised_750例真菌预测孕期.dta')
# check there is no NA in outcome columes
if data['outcome'].isnull().any(): 
    data.dropna(subset=['outcome'], inplace=True)
data.outcome.isnull().any()
data.dtypes


data['id'] = data['ID'].apply(lambda x: x[3:])
input_data = id_kfold(data=data, id=data.id, folds=folds, seed=seed)
input_data.dtypes



## ---------------------------------------------------- original metabolite data -------------------------------------------------------------------------

"""  

input feature: 

outcome: original metabolite data (no-Z-score )

form: continuous

strategy: 10-fold cross validation
 
""" 

input_data.columns
target = 'outcome' ## 根据input修改

target_dia = input_data.outcome # set outcome 
feature_dia =input_data.iloc[:, 5:input_data.shape[1]] 

x_train=feature_dia
y_train=target_dia

model=lgb.LGBMClassifier(**lgb_params)
# kf = KFold(n_splits=folds, shuffle=True, random_state=seed)  # 当shuffle=True时，数据将在划分之前被随机打乱顺序

trn = x_train
val = y_train


tests = []
preds = []
preds_bina = []
meanSHAP = pd.DataFrame() 
shap_fold = pd.DataFrame()


i = 0
for i in range(folds):
    x1, x2 = trn.loc[~(input_data['kfold'] == i), :], trn.loc[input_data['kfold'] == i, :]
    y1, y2 = val.loc[~(input_data['kfold'] == i)], val.loc[input_data['kfold'] == i] 

    model.fit(x1, y1)
    pred = model.predict_proba(x2)[:,1]
    pred2 = model.predict(x2)
    
    tests.append(y2)
    preds.append(pred)
    preds_bina.append(pred2)
    
    # SHAP Model interpreting ------------------------#
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x2)
    shap_values_2d = pd.DataFrame(shap_values[0])  
    shap_values_2d.columns = trn.columns 
    shap_fold = pd.concat([shap_fold, shap_values_2d], axis=0) # 按行合并 
    
    i+=1 
    

# calculate over-all mean SHAP values------------------------- 
shap_abs = abs(shap_fold)
shap_mean = pd.DataFrame(shap_abs.mean())
shap_mean.columns = [seed]
# meanSHAP_all = pd.concat([meanSHAP_all, shap_mean], axis=1)
meanSHAP_all = shap_mean


# output true and predicted value -------------------------------------

output_test = pd.DataFrame()
for i in np.arange(0,folds):
    tests[i] = pd.DataFrame(tests[i])
    output_test = pd.concat([output_test, tests[i]])

nrow = output_test.iloc[:,0].size
output_test.index = np.arange(0,nrow)
output_test.columns = ['true']


output_pred = pd.DataFrame()
for i in np.arange(0,folds):
    preds[i] = pd.DataFrame(preds[i])
    output_pred = pd.concat([output_pred, preds[i]], axis=0)

nrow = output_pred.iloc[:,0].size
output_pred.index = np.arange(0,nrow)  
output_pred.columns = ['pred']


output_pred2 = pd.DataFrame()
for i in np.arange(0,folds):
    preds_bina[i] = pd.DataFrame(preds_bina[i])
    output_pred2 = pd.concat([output_pred2, preds_bina[i]], axis=0)

nrow = output_pred2.iloc[:,0].size
output_pred2.index = np.arange(0,nrow)  
output_pred2.columns = ['pred']


output_total = []
output_total = pd.concat([output_pred, output_test], axis=1)
output_total.columns = [('pred_' + str(seed) ),('test_' + str(seed))]
output_prediction_all = output_total
# output_prediction_all = pd.concat([output_prediction_all, output_total], axis=1)



# evaluate performance and output---------------------------------
model_eval = pd.DataFrame()

fpr, tpr, thresholds = metrics.roc_curve(output_test, output_pred, pos_label=3)
model_eval.loc[seed, 'AUC'] = metrics.auc(fpr, tpr)

accuracy = accuracy_score(output_test, output_pred2)
model_eval.loc[seed, 'Accuracy'] = accuracy 

model_eval_all = model_eval
# pd.concat([model_eval_all, model_eval], axis=0)



# save results----------------------------------
os.chdir(root)
# os.chdir('//172.16.55.11/zhangke/Projects/y_Cooperation/Gestation_prediction_byFuYuanqing/') #修改工作路径

filename = (target + '_fungi750_10fold_' + date + '.xlsx')
writer = pd.ExcelWriter(filename)

model_eval_all.to_excel(writer, sheet_name='performance')
meanSHAP_all.to_excel(writer, sheet_name='meanSHAP') 
output_prediction_all.to_excel(writer, sheet_name='prediction') 

writer._save()

















# Parameters settngs
# -----------------------------------------------------------------------------

root = '/Users/zhangke/Desktop/result_prediction_240313'
os.chdir(root)

# random_seedlist = pd.read_excel('lightGBM_random_seedlist.xlsx') 
seed = 123456
date = '240314'


# set hyperparameters

lgb_params = {
              'objective': 'binary',
              'metric': 'auc',
              'learning_rate': 0.005,
              'feature_fraction': 0.2,
              'min_data_in_leaf': 15, 
              'early_stopping_rounds': None, 'n_estimators': 2000,
              'bagging_fraction': 0.8, 'bagging_freq': 1,
              'num_threads': 1, 'verbose': -1, 'silent': True}


folds = 10
n_bs = 1



data = pd.read_stata('ITS_revised_750例真菌预测孕早中期.dta')
# check there is no NA in outcome columes
if data['outcome'].isnull().any(): 
    data.dropna(subset=['outcome'], inplace=True)
data.outcome.isnull().any()
data.dtypes


data['id'] = data['ID'].apply(lambda x: x[3:])
input_data = id_kfold(data=data, id=data.id, folds=folds, seed=seed)
input_data.dtypes



## ---------------------------------------------------- original metabolite data -------------------------------------------------------------------------

"""  

input feature: 

outcome: original metabolite data (no-Z-score )

form: continuous

strategy: 10-fold cross validation
 
""" 

input_data.columns
target = 'outcome' ## 根据input修改

target_dia = input_data.outcome # set outcome 
feature_dia =input_data.iloc[:, 5:input_data.shape[1]] 

x_train=feature_dia
y_train=target_dia

model=lgb.LGBMClassifier(**lgb_params)
# kf = KFold(n_splits=folds, shuffle=True, random_state=seed)  # 当shuffle=True时，数据将在划分之前被随机打乱顺序

trn = x_train
val = y_train


tests = []
preds = []
preds_bina = []
meanSHAP = pd.DataFrame() 
shap_fold = pd.DataFrame()


i = 0
for i in range(folds):
    x1, x2 = trn.loc[~(input_data['kfold'] == i), :], trn.loc[input_data['kfold'] == i, :]
    y1, y2 = val.loc[~(input_data['kfold'] == i)], val.loc[input_data['kfold'] == i] 

    model.fit(x1, y1)
    pred = model.predict_proba(x2)[:,1]
    pred2 = model.predict(x2)
    
    tests.append(y2)
    preds.append(pred)
    preds_bina.append(pred2)
    
    # SHAP Model interpreting ------------------------#
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x2)
    shap_values_2d = pd.DataFrame(shap_values[0])  
    shap_values_2d.columns = trn.columns 
    shap_fold = pd.concat([shap_fold, shap_values_2d], axis=0) # 按行合并 
    
    i+=1 
    

# calculate over-all mean SHAP values------------------------- 
shap_abs = abs(shap_fold)
shap_mean = pd.DataFrame(shap_abs.mean())
shap_mean.columns = [seed]
# meanSHAP_all = pd.concat([meanSHAP_all, shap_mean], axis=1)
meanSHAP_all = shap_mean


# output true and predicted value -------------------------------------

output_test = pd.DataFrame()
for i in np.arange(0,folds):
    tests[i] = pd.DataFrame(tests[i])
    output_test = pd.concat([output_test, tests[i]])

nrow = output_test.iloc[:,0].size
output_test.index = np.arange(0,nrow)
output_test.columns = ['true']


output_pred = pd.DataFrame()
for i in np.arange(0,folds):
    preds[i] = pd.DataFrame(preds[i])
    output_pred = pd.concat([output_pred, preds[i]], axis=0)

nrow = output_pred.iloc[:,0].size
output_pred.index = np.arange(0,nrow)  
output_pred.columns = ['pred']


output_pred2 = pd.DataFrame()
for i in np.arange(0,folds):
    preds_bina[i] = pd.DataFrame(preds_bina[i])
    output_pred2 = pd.concat([output_pred2, preds_bina[i]], axis=0)

nrow = output_pred2.iloc[:,0].size
output_pred2.index = np.arange(0,nrow)  
output_pred2.columns = ['pred']


output_total = []
output_total = pd.concat([output_pred, output_test], axis=1)
output_total.columns = [('pred_' + str(seed) ),('test_' + str(seed))]
output_prediction_all = output_total
# output_prediction_all = pd.concat([output_prediction_all, output_total], axis=1)



# evaluate performance and output---------------------------------
model_eval = pd.DataFrame()

fpr, tpr, thresholds = metrics.roc_curve(output_test, output_pred, pos_label=3)
model_eval.loc[seed, 'AUC'] = metrics.auc(fpr, tpr)

accuracy = accuracy_score(output_test, output_pred2)
model_eval.loc[seed, 'Accuracy'] = accuracy 

model_eval_all = model_eval
# pd.concat([model_eval_all, model_eval], axis=0)



# save results----------------------------------
os.chdir(root)
# os.chdir('//172.16.55.11/zhangke/Projects/y_Cooperation/Gestation_prediction_byFuYuanqing/') #修改工作路径

filename = (target + '_fungi750_10fold_T2T1_' + date + '.xlsx')
writer = pd.ExcelWriter(filename)

model_eval_all.to_excel(writer, sheet_name='performance')
meanSHAP_all.to_excel(writer, sheet_name='meanSHAP') 
output_prediction_all.to_excel(writer, sheet_name='prediction') 

writer._save()






















# Parameters settngs
# -----------------------------------------------------------------------------

root = '/Users/zhangke/Desktop/result_prediction_240313'
os.chdir(root)

# random_seedlist = pd.read_excel('lightGBM_random_seedlist.xlsx') 
seed = 123456
date = '240314'


# set hyperparameters

lgb_params = {
              'objective': 'binary',
              'metric': 'auc',
              'learning_rate': 0.005,
              'feature_fraction': 0.2,
              'min_data_in_leaf': 15, 
              'early_stopping_rounds': None, 'n_estimators': 2000,
              'bagging_fraction': 0.8, 'bagging_freq': 1,
              'num_threads': 1, 'verbose': -1, 'silent': True}


folds = 10
n_bs = 1



data = pd.read_stata('ITS_revised_750例真菌预测孕中晚期.dta')
# check there is no NA in outcome columes
if data['outcome'].isnull().any(): 
    data.dropna(subset=['outcome'], inplace=True)
data.outcome.isnull().any()
data.dtypes


data['id'] = data['ID'].apply(lambda x: x[3:])
input_data = id_kfold(data=data, id=data.id, folds=folds, seed=seed)
input_data.dtypes



## ---------------------------------------------------- original metabolite data -------------------------------------------------------------------------

"""  

input feature: 

outcome: original metabolite data (no-Z-score )

form: continuous

strategy: 10-fold cross validation
 
""" 

input_data.columns
target = 'outcome' ## 根据input修改

target_dia = input_data.outcome # set outcome 
feature_dia =input_data.iloc[:, 5:input_data.shape[1]] 

x_train=feature_dia
y_train=target_dia

model=lgb.LGBMClassifier(**lgb_params)
# kf = KFold(n_splits=folds, shuffle=True, random_state=seed)  # 当shuffle=True时，数据将在划分之前被随机打乱顺序

trn = x_train
val = y_train


tests = []
preds = []
preds_bina = []
meanSHAP = pd.DataFrame() 
shap_fold = pd.DataFrame()


i = 0
for i in range(folds):
    x1, x2 = trn.loc[~(input_data['kfold'] == i), :], trn.loc[input_data['kfold'] == i, :]
    y1, y2 = val.loc[~(input_data['kfold'] == i)], val.loc[input_data['kfold'] == i] 

    model.fit(x1, y1)
    pred = model.predict_proba(x2)[:,1]
    pred2 = model.predict(x2)
    
    tests.append(y2)
    preds.append(pred)
    preds_bina.append(pred2)
    
    # SHAP Model interpreting ------------------------#
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x2)
    shap_values_2d = pd.DataFrame(shap_values[0])  
    shap_values_2d.columns = trn.columns 
    shap_fold = pd.concat([shap_fold, shap_values_2d], axis=0) # 按行合并 
    
    i+=1 
    

# calculate over-all mean SHAP values------------------------- 
shap_abs = abs(shap_fold)
shap_mean = pd.DataFrame(shap_abs.mean())
shap_mean.columns = [seed]
# meanSHAP_all = pd.concat([meanSHAP_all, shap_mean], axis=1)
meanSHAP_all = shap_mean


# output true and predicted value -------------------------------------

output_test = pd.DataFrame()
for i in np.arange(0,folds):
    tests[i] = pd.DataFrame(tests[i])
    output_test = pd.concat([output_test, tests[i]])

nrow = output_test.iloc[:,0].size
output_test.index = np.arange(0,nrow)
output_test.columns = ['true']


output_pred = pd.DataFrame()
for i in np.arange(0,folds):
    preds[i] = pd.DataFrame(preds[i])
    output_pred = pd.concat([output_pred, preds[i]], axis=0)

nrow = output_pred.iloc[:,0].size
output_pred.index = np.arange(0,nrow)  
output_pred.columns = ['pred']


output_pred2 = pd.DataFrame()
for i in np.arange(0,folds):
    preds_bina[i] = pd.DataFrame(preds_bina[i])
    output_pred2 = pd.concat([output_pred2, preds_bina[i]], axis=0)

nrow = output_pred2.iloc[:,0].size
output_pred2.index = np.arange(0,nrow)  
output_pred2.columns = ['pred']


output_total = []
output_total = pd.concat([output_pred, output_test], axis=1)
output_total.columns = [('pred_' + str(seed) ),('test_' + str(seed))]
output_prediction_all = output_total
# output_prediction_all = pd.concat([output_prediction_all, output_total], axis=1)



# evaluate performance and output---------------------------------
model_eval = pd.DataFrame()

fpr, tpr, thresholds = metrics.roc_curve(output_test, output_pred, pos_label=3)
model_eval.loc[seed, 'AUC'] = metrics.auc(fpr, tpr)

accuracy = accuracy_score(output_test, output_pred2)
model_eval.loc[seed, 'Accuracy'] = accuracy 

model_eval_all = model_eval
# pd.concat([model_eval_all, model_eval], axis=0)



# save results----------------------------------
os.chdir(root)
# os.chdir('//172.16.55.11/zhangke/Projects/y_Cooperation/Gestation_prediction_byFuYuanqing/') #修改工作路径

filename = (target + '_fungi750_10fold_T3T2_' + date + '.xlsx')
writer = pd.ExcelWriter(filename)

model_eval_all.to_excel(writer, sheet_name='performance')
meanSHAP_all.to_excel(writer, sheet_name='meanSHAP') 
output_prediction_all.to_excel(writer, sheet_name='prediction') 

writer._save()






