# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print('main triggered')
# import os

# os.getcwd()
# os.chdir('Data_EMC')
# os.chdir('C:\\Users\\777018\\Documents\\data_EMC')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime

import random

from functions import *
from classes import *

# inputs = ['Data_EMC/labs_up_to_date.csv','Data_EMC/20201102_covid data_metingen_alive.csv','Data_EMC/20201102_covid data_metingen_dead.csv']

# inputs = ['D:/Data_EMC/20201116_covid_data_lab.csv','D:/Data_EMC/20201115_covid_data_metingen.csv']
inputs = ['../../Data_EMC/20210112_covid data_lab.csv','../../Data_EMC/20210111_covid data_metingen.xlsx','../../Data_EMC/pirate.xlsx']
# inputs = ['../Data_EMC/20201130_covid_data_lab.csv','../Data_EMC/20201123_covid data_metingen.csv','../Data_EMC/pirate.xlsx']
encoders = ['utf-8',"ISO-8859-1",'ascii']

# Model characteristics
pred_window = 24
gap = 0
feature_window = 1
label_type = 'ICU'
model = 'RF'
knn = 2
n_trees = 500
recall_threshold = 0.8
FS = False
n_keep = 50


# prints
prints_to_text = True
save_model= True
save_results_dir = 'results_test'

k = 10

# Input characteristics
n_features = 20
n_demo = 2

#sampling strategy
int_neg = 12
int_pos = 12

# CV
val_share = 0.4
test_share = 0.2

# flags
policy = True
freq = False
time=True
inter = False
stats = True
diff = True
sliding_window = 24
balance=False


# make tuple with model specs
specs = dict({'n_features':n_features,'pred_window':pred_window,'gap':gap,'feature_window':feature_window,
              'int_neg':int_neg,'int_pos':int_pos,'val_share':val_share,'test_share':test_share,
              'label_type':label_type,'model':model,'policy':policy,'freq':freq,'time':time,
              'inter':inter,'stats':stats,'diff':diff,'sliding_window':sliding_window,'n_demo':n_demo,
              'balance':balance,'knn':knn,'n_trees':n_trees,'recall_threshold':recall_threshold,'FS':FS,
             'save_results':save_results_dir})




print('RESULTS FOR MODEL:',model)
print('PRED W:',pred_window,'Hours')
print('FEATURE W:',feature_window, 'samples')
if freq:
    print('Feature freq as extra features')
if time:
    print('LOS (in days) as extra features')
if policy:
    print('No IC policy patients filtered')
#%%
parchure = Parchure(specs)

# # parchure.import_pacmed(n_features)  # Pacmed data, Feature selection in this func

parchure.import_cci(inputs)

data, dict_unit = parchure.import_labs(inputs,encoders)  # EMC data
parchure.clean_labs()
parchure.import_vitals(inputs,encoders)
parchure.clean_vitals()
df,ids_events,ids_clinic = parchure.merge()    #feature selection in this func


# parchure.Optimize_trees()



#%%
df_pos = pd.DataFrame()
for i in ids_events:
    ex = df[df['ID']==i].sort_values(by='TIME')
    df_pos = pd.concat([df_pos,ex],axis=0)
    

df_neg = pd.DataFrame()
for i in ids_clinic:
    ex = df[df['ID']==i].sort_values(by='TIME')
    df_neg = pd.concat([df_neg,ex],axis=0)
    


demo_pos = list()

for i in np.unique(df_pos['ID']):
    demo = list()
    demo.append(i)
    
    ex = df_pos[df_pos['ID']==i]
    
    if ex['BMI'].dropna().shape[0]<1:
        demo.append(np.nan)
    else:
        demo.append(np.round(float(ex['BMI'].dropna().min()),1))
    
    if ex['AGE'].dropna().shape[0]<1:
        demo.append(np.nan)
    else:
        demo.append(int(ex['AGE'].dropna().min()))
    
    t_event = ex[ex['DEPARTMENT']=='IC']['START'].min()
    demo.append(np.round((t_event-ex['TIME'].min()).total_seconds()/3600,0))
    
    demo_pos.append(demo)
    
demo_pos = pd.DataFrame(demo_pos,columns=['ID','BMI','AGE','LOS'])
mask = demo_pos['LOS']<0
demo_pos = demo_pos[~mask]

def make_table(df):
    ex = list()
    ex.append(str(np.round(np.mean(df.AGE),1))+' ('+str(np.round(np.std(df.AGE),1))+')')
    ex.append(str(np.round(np.median(df.dropna().AGE),1))+' ('+str(np.round(np.min(df.dropna().AGE),1))+','+str(np.round(np.max(df.dropna().AGE),1)) +')')
    n = sum((df.AGE>=18) & (df.AGE<=45))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>45) & (df.AGE<=65))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>65) & (df.AGE<=80))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>80))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum(df.AGE.isna())
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    
    ex.append(str(np.round(np.mean(df.dropna().BMI),1))+' ('+str(np.round(np.std(df.dropna().BMI),1))+')')
    ex.append(str(np.round(np.median(df.dropna().BMI),1))+' ('+str(np.round(np.min(df.dropna().BMI),1))+','+str(np.round(np.max(df.dropna().BMI),1)) +')')
    
    n = sum((df.LOS<=24))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>24) & (df.LOS<=72))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>72) & (df.LOS<=240))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>240))
    ex.append(str(n) + '('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
        
    return ex

pos_ex = make_table(demo_pos)
pos_ex = pd.DataFrame(pos_ex)
pos_ex.to_excel('demo_pos.xlsx')

demo_neg = list()

for i in np.unique(df_neg['ID']):
    demo = list()
    demo.append(i)
    
    ex = df_neg[df_neg['ID']==i]
    
    if ex['BMI'].dropna().shape[0]<1:
        demo.append(np.nan)
    else:
        demo.append(np.round(float(ex['BMI'].dropna().min()),1))
    
    if ex['AGE'].dropna().shape[0]<1:
        demo.append(np.nan)
    else:
        demo.append(int(ex['AGE'].dropna().min()))
    
    if ex['DISCHARGE'].dropna().shape[0]<1:
            demo.append(np.round((ex['TIME'].max()-ex['TIME'].min()).total_seconds()/3600,0))
    else:
        demo.append(np.round((ex['DISCHARGE'].min()-ex['TIME'].min()).total_seconds()/3600,0))
    
    demo_neg.append(demo)
    
demo_neg = pd.DataFrame(demo_neg,columns=['ID','BMI','AGE','LOS'])
mask = demo_neg['LOS']<0
demo_neg = demo_neg[~mask]

neg_ex = make_table(demo_neg)
neg_ex = pd.DataFrame(neg_ex)
neg_ex.to_excel('demo_neg.xlsx')  
        
        

#%%
# df_missing,n_clinic,n_event = parchure.missing(x_days=False)

# import seaborn as sns

# df_missing.loc[:,'/day'] = df_missing['/day'].astype(float)
# order = df_missing['feature'].unique()[5:25]

# df = df_missing[df_missing['feature'].isin(order)]

# # Draw a nested boxplot to show bills by day and time
# plt.figure()
# ax = sns.boxplot(x="feature", y="/day",
#             hue="label", palette=["m", "g"],
#             data=df,order=order)

# ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# plt.tight_layout()
# plt.savefig('freqs_labs',dpi=200)

# df_days = pd.DataFrame()
# df_days['LOS'] = n_clinic
# df_days['group'] = 'clinic_only'
# df_days_2 = pd.DataFrame()
# df_days_2['LOS'] = n_event
# df_days_2['group'] = 'clinic_to_ICU'
# df_days = pd.concat([df_days,df_days_2],axis=0)

# # Draw a nested boxplot to show bills by day and time
# plt.figure()
# ax_2 = sns.boxplot(x='group', y="LOS",
#               palette=["m", "g"],
#             data=df_days)

# # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
# plt.tight_layout()
# plt.savefig('Clinic_days',dpi=200)


#%%


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

f, axes = plt.subplots(2, 2, figsize=(10, 10))


y_real = []
y_proba = []
X_n = []
y_proba_na = []



aps_NEWS = []
aps_model = []
aucs_NEWS = []
aucs_model = []

for i in range(k):
    
    print('------ Fold',i,' -----')
    
    # ----- Train / val  / test split and normalization ---------
    val,val_raw,train,train_raw,test,test_raw = parchure.Prepare(random.randint(0, 10000))
    
    # ----- Build feature vectors -----
    imputer, imputer_raw,pos_tot,neg_tot = parchure.Build_feature_vectors(i,str(model)+'_'+str(n_features)) 
    
    if i == 0:
        selector = parchure.feature_selection(n_keep)
       
    # --------- optimize weight for minority class ------
    t = parchure.Optimize_weights()
    
    # ---- Get stats for this fold -------
    clf, explainer,df_val,median,df_demo_val,demo_median,df_val_raw,median_raw,df_demo_val_raw,demo_median_raw,precision,recall,ap,fpr,tpr,auc,ytest,pred,precision_n,recall_n,fpr_n,ap_n,auc_n,X,pred_na,X_val = parchure.Predict()
    aps_NEWS.append(ap_n)
    aps_model.append(ap)
    aucs_NEWS.append(auc_n)
    aucs_model.append(auc)
    
    # ------ Update best fold --------
    if (i == 0) or (ap>ap_best):
        ap_best = ap
        clf_best = clf
        explainer_best = explainer
        
        df_val_best = df_val
        median_best = median
        df_demo_val_best = df_demo_val
        demo_median_best = demo_median
        
        df_val_raw_best = df_val_raw
        median_raw_best = median_raw
        df_demo_val_raw_best = df_demo_val_raw
        demo_median_raw_best = demo_median_raw
        
        X_val_best = X_val
        
        imputer_best = imputer
        imputer_raw_best = imputer_raw
        
        t_best = t
        print('Updated best AP:',ap_best)
    else:
        print('best AP:', ap_best)
            
        
    
    #Dummy
    y_proba_na.append(pred_na)
    
    #NEWS
    lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap_n,3))
    axes[0,0].step(recall_n, precision_n, label=lab)
    lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc_n,3))
    axes[0,1].step(fpr_n, recall_n, label=lab)
    X_n.append(X)


    #Model
    lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap,3))
    axes[1,0].step(recall, precision, label=lab)
    lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc,3))
    axes[1,1].step(fpr, tpr, label=lab)
    y_real.append(ytest)
    y_proba.append(pred)




if prints_to_text:
    import sys
    sys.stdout = open('results/results_'+str(pred_window)+'.txt','wt')
    
# --- Save model and attributes for later analysis---
if save_model:
    import pickle
    filename = 'saved_model/trained_model.sav'
    pickle.dump(clf_best, open(filename, 'wb'))
    
    filename = 'saved_model/explainer_best.sav'
    pickle.dump(explainer_best, open(filename, 'wb'))
    
    filename = 'saved_model/imputer_best.sav'
    pickle.dump(imputer_best, open(filename, 'wb'))
    
    filename = 'saved_model/imputer_raw_best.sav'
    pickle.dump(imputer_raw_best, open(filename, 'wb'))
    
    if FS:
        filename = 'saved_model/selector.sav'
        pickle.dump(selector, open(filename, 'wb'))
    
    pd.DataFrame(df_val_best).to_csv('saved_model/df_val_best.csv')
    pd.DataFrame(median_best).to_csv('saved_model/median_best.csv')
    pd.DataFrame(df_demo_val_best).to_csv('saved_model/df_demo_val_best.csv')
    pd.DataFrame(demo_median_best).to_csv('saved_model/demo_median_best.csv')
    
    pd.DataFrame(df_val_raw_best).to_csv('saved_model/df_val_raw_best.csv')
    pd.DataFrame(median_raw_best).to_csv('saved_model/median_raw_best.csv')
    pd.DataFrame(df_demo_val_raw_best).to_csv('saved_model/df_demo_val_raw_best.csv')
    pd.DataFrame(demo_median_raw_best).to_csv('saved_model/demo_median_raw_best.csv')
    pd.DataFrame(X_val_best).to_csv('saved_model/X_val_best.csv')

aps_NEWS = np.asarray(aps_NEWS)
aps_model = np.asarray(aps_model)
aucs_NEWS = np.asarray(aucs_NEWS)
aucs_model = np.asarray(aucs_model)


y_real_full = np.concatenate(y_real)
y_proba_na_full = np.concatenate(y_proba_na)
X_n_full = np.concatenate(X_n)
y_proba_full = np.concatenate(y_proba)


# ---------- NEWS ---------------------
precision, recall,fpr,thresholds = results_news(X_n_full,y_real_full,'threshold')
lab = 'Overall AP=%.4f' % (np.round(AP_manually(precision, recall),3))
axes[0,0].step(recall, precision, label=lab, lw=2, color='black')

precision_na, recall_na, _ = precision_recall_curve(y_real_full, y_proba_na_full)
lab = 'Dummy AP=%.4f' % np.round(average_precision_score(y_real_full, y_proba_na_full),3) 
axes[0,0].step(recall_na, precision_na, label=lab, lw=2, color='grey')

axes[0,0].set_xlabel('Recall')
axes[0,0].set_ylabel('Precision')
axes[0,0].legend(loc='upper right', fontsize='small')
axes[0,0].set_title('PR-curve NEWS')


auc = metrics.auc(fpr, recall)
lab = 'Overall AUC=%.4f' % (np.round(auc,3))
axes[0,1].step(fpr, recall, label=lab, lw=2, color='black')

axes[0,1].set_xlabel('FPR')
axes[0,1].set_ylabel('Recall')
axes[0,1].legend(loc='lower right', fontsize='small')
axes[0,1].set_title('ROC NEWS')



print('ap NEWS:', aps_NEWS,' \n mean:',np.mean(aps_NEWS), ' \n std:', np.std(aps_NEWS))
print('ap Model:', aps_model,' \n mean:',np.mean(aps_model), ' \n std:', np.std(aps_model))
print('auc NEWS:', aucs_NEWS,' \n mean:',np.mean(aucs_NEWS), ' \n std:', np.std(aucs_NEWS))
print('auc Model:', aucs_model,' \n mean:',np.mean(aucs_model), ' \n std:', np.std(aucs_model))


#Overall confusion matrix NEWS
print("OVERALL STATS NEWS")


#leave out last value for precision(=1) and recall (=0)
precision = precision[:-1]
recall = recall[:-1]
thresholds = thresholds[:-1]
    
betas = [2,3,4,5,6,7,8,9,10]
for beta in betas:
    print('\n --- results for Beta = ',beta, '--------')
    
    f2_scores = (1+beta**2)*recall*precision/(recall+(beta**2*precision))    
    idx = np.argwhere(np.isnan(f2_scores))
    f2_scores = np.delete(f2_scores, idx)
    thresholds = np.delete(thresholds, idx)
    t = thresholds[np.argmax(f2_scores)]
    preds = (y_proba_full>t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real_full, preds).ravel()
    
    print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
    print('sens:',np.round(tp/(tp+fn),2),'spec:',np.round(tn/(tn+fp),2))
    r = np.round(tp/(tp+fn),2)
    p = np.round(tp/(tp+fp),2)
    print('Recall:',r,'Pecision:',p)
    

# ------------- Model ----------------------
precision, recall, thresholds = precision_recall_curve(y_real_full, y_proba_full)
lab = 'Overall AP=%.4f' % np.round(average_precision_score(y_real_full, y_proba_full),3)
axes[1,0].step(recall, precision, label=lab, lw=2, color='black')

precision_na, recall_na, _ = precision_recall_curve(y_real_full, y_proba_na_full)
lab = 'Dummy AP=%.4f' % np.round(average_precision_score(y_real_full, y_proba_na_full),3) 
axes[1,0].step(recall_na, precision_na, label=lab, lw=2, color='grey')

axes[1,0].set_xlabel('Recall')
axes[1,0].set_ylabel('Precision')
axes[1,0].legend(loc='upper right', fontsize='small')
axes[1,0].set_title('PR-curve Model')

fpr, tpr, _ = metrics.roc_curve(y_real_full, y_proba_full)
auc = metrics.auc(fpr, tpr)
lab = 'Overall AUC=%.4f' % np.round(auc,3)
axes[1,1].step(fpr, tpr, label=lab, lw=2, color='black')

axes[1,1].set_xlabel('FPR')
axes[1,1].set_ylabel('Recall')
axes[1,1].legend(loc='lower right', fontsize='small')
axes[1,1].set_title('ROC Model')


f.tight_layout()
if save_model:
    f.savefig('results/result.png',dpi=300)


#Overall confusion matrix MODEL
print('\n OVERALL STATS MODEL')

#leave out last value for precision(=1) and recall (=0)
precision = precision[:-1]
recall = recall[:-1]


betas = [2,3,4,5,6,7,8,9,10]
betas_POC = []
for beta in betas:
    print('\n --- results for Beta = ',beta, '--------')
    
    f2_scores = (1+beta**2)*recall*precision/(recall+(beta**2*precision))    
    idx = np.argwhere(np.isnan(f2_scores))
    f2_scores = np.delete(f2_scores, idx)
    thresholds = np.delete(thresholds, idx)
    t = thresholds[np.argmax(f2_scores)]
    print('optimal threshold:',t)
    preds = (y_proba_full>t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real_full, preds).ravel()
    print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
    print('sens:',np.round(tp/(tp+fn),2),'spec:',np.round(tn/(tn+fp),2))
    r = np.round(tp/(tp+fn),2)
    p = np.round(tp/(tp+fp),2)
    print('Recall:',r,'Pecision:',p)
    if np.round(tp/(tp+fn),2) > specs['recall_threshold']:
        betas_POC.append(beta)

        
betas_POC = np.asarray(betas_POC)
print('Beta values which guarantee'+ str(specs['recall_threshold'])+' sensitivity:', betas_POC)


if len(betas_POC) < 1:
    beta = 2
    print('Something up with betas, do run with beta = 2')
else:
    beta=np.min(betas_POC)
print('----- START POC plots with best threshold:',t_best)
print('idxs for fetaures to keep:',top_idx)
#%%

# parchure.Prepare(random.randint(0, 100),val_share=val_share) # Train / val  / test split and normalization

# parchure.Build_feature_vectors(1,pred_window,gap,int_neg,int_pos,feature_window,str(model)+'_'+str(n_features),freq=freq,time=time) 
# # parchure.Balance(undersampling=True)

# w = parchure.Optimize()
# parchure.Train(w,model=model,balance=False)

# auc,ap,tn, fp, fn, tp = parchure.Evaluate(feature_window,2)

#%% load data


# #Saved direcertory
# d = 'saved_model_08_01'

# import pickle
# with open(d+'/trained_model.sav', 'rb') as file:
#     clf_best = pickle.load(file)
# with open(d +'/explainer_best.sav', 'rb') as file:
#     explainer_best = pickle.load(file)

# with open(d +'/imputer_best.sav', 'rb') as file:
#     imputer_best = pickle.load(file)
# with open(d +'/imputer_raw_best.sav', 'rb') as file:
#     imputer_raw_best = pickle.load(file)

# type_lib = {'ID':str,
#                 'VARIABLE':str,
#                 'TIME': str,
#                 'VALUE': float,
#                 'DEPARTMENT':str,
#                 'AGE':float,
#                 'BMI':float,
                
#                 }


# df_val_best = pd.read_csv(d+'/df_val_best.csv',index_col=False,dtype = type_lib)
# df_val_best = df_val_best.drop(df_val_best.columns[0], axis=1)

# median_best = pd.read_csv(d +'/median_best.csv',index_col=False,dtype = type_lib)
# median_best = median_best.drop(median_best.columns[0], axis=1)
# median_best = np.asarray(median_best)

# df_demo_val_best = pd.read_csv(d +'/df_demo_val_best.csv',index_col=False,dtype = type_lib)
# df_demo_val_best = df_demo_val_best.drop(df_demo_val_best.columns[0], axis=1)

# demo_median_best = pd.read_csv(d +'/demo_median_best.csv',index_col=False,dtype = type_lib)
# demo_median_best = demo_median_best.drop(demo_median_best.columns[0], axis=1)
# demo_median_best = np.asarray(demo_median_best)

# df_val_raw_best = pd.read_csv(d +'/df_val_raw_best.csv',index_col=False,dtype = type_lib)
# df_val_raw_best = df_val_raw_best.drop(df_val_raw_best.columns[0], axis=1)

# median_raw_best = pd.read_csv(d +'/median_raw_best.csv',index_col=False,dtype = type_lib)
# median_raw_best = median_raw_best.drop(median_raw_best.columns[0], axis=1)
# median_raw_best = np.asarray(median_raw_best)

# df_demo_val_raw_best = pd.read_csv(d +'/df_demo_val_raw_best.csv',index_col=False,dtype = type_lib)
# df_demo_val_raw_best = df_demo_val_raw_best.drop(df_demo_val_raw_best.columns[0], axis=1)

# demo_median_raw_best = pd.read_csv(d +'/demo_median_raw_best.csv',index_col=False,dtype = type_lib)
# demo_median_raw_best = demo_median_raw_best.drop(demo_median_raw_best.columns[0], axis=1)
# demo_median_raw_best = np.asarray(demo_median_raw_best)

# date_format='%Y-%m-%d %H:%M:%S'

# df_val_best['TIME'] = pd.to_datetime(df_val_best['TIME'],format = date_format)
# df_val_raw_best['TIME'] = pd.to_datetime(df_val_raw_best['TIME'],format = date_format)



#%% Proof of concept

from functions import *
from classes import *


X_val_pos = parchure.Proof_of_concept(clf_best,explainer_best,imputer_best,imputer_raw_best,
                                        df_val_best,median_best,df_demo_val_best,demo_median_best,
                                        df_val_raw_best,median_raw_best,df_demo_val_raw_best,demo_median_raw_best,
                                        t_best,1,'pos',plot=False)

X_val_neg = parchure.Proof_of_concept(clf_best,explainer_best,imputer_best,imputer_raw_best,
                                        df_val_best,median_best,df_demo_val_best,demo_median_best,
                                        df_val_raw_best,median_raw_best,df_demo_val_raw_best,demo_median_raw_best,
                                        t_best,1,'neg',plot=False)

X_val_tot = np.concatenate([X_val_pos,X_val_neg],axis=0)

parchure.Global_Feature_importance(X_val_tot,explainer_best,clf_best)

# X_val_pos = parchure.Proof_of_concept(clf_best,explainer_best,imputer_best,imputer_raw_best,
#                                         df_val_best,median_best,df_demo_val_best,demo_median_best,
#                                         df_val_raw_best,median_raw_best,df_demo_val_raw_best,demo_median_raw_best,
#                                         t_best,1,'pos',plot=True)

# X_val_neg = parchure.Proof_of_concept(clf_best,explainer_best,imputer_best,imputer_raw_best,
#                                         df_val_best,median_best,df_demo_val_best,demo_median_best,
#                                         df_val_raw_best,median_raw_best,df_demo_val_raw_best,demo_median_raw_best,
#                                         t_best,1,'neg',plot=True)

#%% Nested CV

# parchure = Parchure(inputs=inputs,encoders=encoders)

# # parchure.import_pacmed(n_features)  # Pacmed data, Feature selection in this func

# parchure.import_labs()  # EMC data
# parchure.clean_labs()
# parchure.import_vitals()
# parchure.clean_vitals()
# parchure.merge(n_features)    #feature selection in this func

# train_aucs = []
# aucs = []
# tns = []
# fps = []
# fns=[]
# tps=[]


# for i in range(10): # 10 fold CV
#     parchure.Prepare(random.randint(0, 10),val_share=val_share) # Train / val  / test split and normalization
#     parchure.Build_feature_vectors(1,pred_window,gap,freq,feature_window,str(model)+'_'+str(n_features)) 
#     parchure.Balance(undersampling=True)
#     parchure.Train(model=model,balance=True)
#     auc,tn, fp, fn, tp = parchure.Evaluate()
    
#     if i ==1:
#         FI,features = parchure.Plot_results(feature_window,model)
            
#     aucs.append(auc)
#     tns.append(tn)
#     fps.append(fp)
#     fns.append(fn)
#     tps.append(tp)
#     print('----------------AUC of CV ',i,':',auc,'---------------------------- \n \n')
#     print('----------------Confusion matrix of CV ',i,':','TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp,'------- \n \n')
    
# print('-------------------')
# print('RESULTS FOR ',model,'predicting',label_type,' \n n lab features:', n_features, '\n features window length:', feature_window, 'samples', 
#       '\n gap:', gap,'hours \n pred window:',pred_window,'hours')

# print('-----Training AUCS----')
# print(train_aucs)
# print('mean training auc:', np.mean(train_aucs))
# print('std:',np.std(train_aucs))
# print('-----Evaluation AUCS----')
# print(aucs)
# print('mean auc:', np.mean(aucs))
# print('std:',np.std(aucs))
# print('mean Confusion matrix:','TN:',np.mean(tns),'FP:',np.mean(fps),'FN:',np.mean(fns),'TP:',np.mean(tps))

# features = np.asarray(['BMI','AGE']+list(features))

# top_5_idx = FI.argsort()[-5:][::-1]
# print('Top 5 features:',features[top_5_idx])



# indices = np.argsort(FI)
# plt.figure()
# plt.title('Feature Importances')
# plt.barh(range(len(features)), FI[indices], color='b', align='center')
# plt.yticks(range(len(FI)), features[indices])
# plt.xlabel('Relative Importance')

# plt.tight_layout()
# plt.savefig('feature_importance_'+str(model)+'_'+str(pred_window),dpi=300)
# # plt.show()

