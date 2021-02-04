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
inputs = ['../../Data_EMC/20201214_covid_data_lab.csv','../../Data_EMC/20201214_covid_data_metingen.csv','../../Data_EMC/pirate.xlsx',
          '../../Data_EMC/MCOLS aanlevering _ 20210126_V3.xlsx']
# inputs = ['../Data_EMC/20201130_covid_data_lab.csv','../Data_EMC/20201123_covid data_metingen.csv','../Data_EMC/pirate.xlsx']
encoders = ['utf-8',"ISO-8859-1",'ascii']

# Model characteristics
pred_window = 24
gap = 0
feature_window = 1
label_type = 'ICU'
model = 'LR'
knn = 2
n_trees = 500
recall_threshold = 0.8
FS = False
n_keep = 50


# prints
prints_to_text = True
save_model= True
save_results_dir = '../MAASSTAD/exc_LOS_LR'
save_model_dir = '../MAASSTAD/exc_LOS_LR'

k = 10

# Input characteristics
n_features = 20
n_demo = 2

#sampling strategy
int_neg = 24
int_pos = 24

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
# parchure = Parchure()
# data = parchure.import_MAASSTAD(inputs,encoders,specs)
# features = parchure.clean_MAASSTAD(specs)
# df_full,ids_events,df_demo = parchure.fix_episodes()
# X_MSD,dens_2,Y_MSD = parchure.Build_feature_vectors(1,str(model)+'_'+str(n_features))


parchure = Parchure()
parchure.import_cci(inputs)
data, dict_unit = parchure.import_labs(inputs,encoders,specs)  # EMC data
parchure.clean_labs()
parchure.import_vitals(inputs,encoders)
parchure.clean_vitals()
df_raw = parchure.merge()    #feature selection in this func
df_full, ids_events,df_demo = parchure.fix_episodes()
X_EMC_LOS,dens_2,Y_EMC_LOS = parchure.Build_feature_vectors(1,str(model)+'_'+str(n_features))

# parchure.Optimize_trees()

# %% TRAIN FULL MODEL

scaler,imputer,clf,explainer = parchure.Train_full_model('LR')


save_model_dir = '../EMC/full_model_LOS_LR'

import pickle
filename = save_model_dir+'/trained_model.sav'
pickle.dump(clf, open(filename, 'wb'))

filename = save_model_dir+'/scaler_best.sav'
pickle.dump(scaler, open(filename, 'wb'))

filename = save_model_dir+'/explainer_best.sav'
pickle.dump(explainer, open(filename, 'wb'))

filename = save_model_dir+'/imputer_best.sav'
pickle.dump(imputer, open(filename, 'wb'))




# %% LOAD MODEL
import pickle
save_model_dir = '../EMC/full_model_LOS_LR'

filename = save_model_dir+'/trained_model.sav'
clf = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/scaler_best.sav'
scaler = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/explainer_best.sav'
explainer = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/imputer_best.sav'
imputer = pickle.load(open(filename, 'rb'))


X = X_MSD[:,:-1]
# X = np.delete(X,1,1)
y = Y_MSD.label

y_pred,X = Predict_full_model(clf,scaler,imputer,X,y)    
from sklearn import datasets, metrics
metrics.plot_roc_curve(clf, X, y)
plt.savefig(save_model_dir + '/ROC_curve.png',dpi=300)
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y, y_pred)
plt.figure()
plt.plot(recall,precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(save_model_dir + '/PR_curve.png',dpi=300)

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


#%% Utility analysis

Y = pd.read_csv('../COVID_PREDICT/Y_covid_predict.csv')
prev = 0.017016317016317017
plot_Utility(Y_MSD,specs,prev)



#%%


# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix

# f, axes = plt.subplots(2, 2, figsize=(10, 10))


# y_real = []
# y_proba = []
# X_n = []
# y_proba_na = []
# y_ts = []
# y_pats = []


# aps_NEWS = []
# aps_model = []
# aucs_NEWS = []
# aucs_model = []



# for i in range(k):
    
#     print('------ Fold',i,' -----')
    
#     imputer_raw,imputer,y_pat,y_t,random_state,scaler = parchure.Prepare(random.randint(0, 10000))
    
#     if i == 0:
#         selector,total_features = parchure.feature_selection(n_keep)
#         print(total_features)
       
#     clf,explainer = parchure.Optimize()
    
#     # ---- Get stats for this fold -------
#     y_true,y_pred,X,X_val = parchure.Predict()
    
    
#     #NEWS
#     precision_n, recall_n,fpr_n,_= results_news(X,y_true,'threshold')
#     auc_n = metrics.auc(fpr_n, recall_n)
#     ap_n = AP_manually(precision_n, recall_n)
    
#     lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap_n,3))
#     axes[0,0].step(recall_n, precision_n, label=lab)
#     lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc_n,3))
#     axes[0,1].step(fpr_n, recall_n, label=lab)

#     X_n.append(X)
#     aps_NEWS.append(ap_n)
#     aucs_NEWS.append(auc_n)
    
#     # MODEL
#     precision, recall, _ = precision_recall_curve(y_true, y_pred)
#     fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
#     auc = metrics.auc(fpr, tpr)
#     ap = average_precision_score(y_true, y_pred)
    
#     lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap,3))
#     axes[1,0].step(recall, precision, label=lab)
#     lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc,3))
#     axes[1,1].step(fpr, tpr, label=lab)
    
#     y_real.append(y_true)
#     y_proba.append(y_pred)
#     y_ts.append(y_t)
#     y_pats.append(y_pat)
#     aps_model.append(ap)
#     aucs_model.append(auc)
    
    
#     if i ==0:
#         if model != 'LR':
#             shap_values = explainer.shap_values(X_val)
#             shap_values_tot = shap_values[1]
#         X_val_tot = X_val
#     else:
#         X_val_tot = np.concatenate([X_val_tot,X_val],axis=0)
#         if model != 'LR':
#             shap_values_tot = np.concatenate([shap_values_tot,explainer.shap_values(X_val)[1]],axis=0)
    
#     # ------ Update best fold --------
#     if (i == 0) or (auc>auc_best):
#         auc_best = auc
#         clf_best = clf
#         scaler_best = scaler
#         explainer_best = explainer
#         random_state_best = random_state
#         imputer_best = imputer
#         imputer_raw_best = imputer_raw
        
#         print('Updated best AUC:',auc_best, ' for validation fold ', i)
        
        
            
        
# print('best AUC found for validation fold:', i,' with: ', auc_best)


# aps_NEWS = np.asarray(aps_NEWS)
# X_n_full = np.concatenate(X_n)
# aucs_NEWS = np.asarray(aucs_NEWS)

# aps_model = np.asarray(aps_model)
# aucs_model = np.asarray(aucs_model)


# y_real_full = np.concatenate(y_real)
# y_proba_full = np.concatenate(y_proba)
# y_t_full = np.concatenate(y_ts)
# y_pat_full = np.concatenate(y_pats)

# Y = pd.DataFrame()
# Y['label'] = y_real_full
# Y['pred'] = y_proba_full
# Y['time_to_event'] = y_t_full
# Y['patient'] = y_pat_full


# assert y_real_full.shape == y_proba_full.shape == y_t_full.shape == y_pat_full.shape

# if prints_to_text:
#     import sys
#     sys.stdout = open(save_results_dir+'/results_'+str(pred_window)+'.txt','wt')
    
# # --- Save model and attributes for later analysis---
# if save_model:
#     import pickle
#     filename = save_model_dir+'/trained_model.sav'
#     pickle.dump(clf_best, open(filename, 'wb'))
    
#     filename = save_model_dir+'/scaler_best.sav'
#     pickle.dump(scaler_best, open(filename, 'wb'))
    
#     filename = save_model_dir+'/explainer_best.sav'
#     pickle.dump(explainer_best, open(filename, 'wb'))
    
#     filename = save_model_dir+'/imputer_best.sav'
#     pickle.dump(imputer_best, open(filename, 'wb'))
    
#     filename = save_model_dir+'/imputer_raw_best.sav'
#     pickle.dump(imputer_raw_best, open(filename, 'wb'))
    
#     if FS:
#         filename = save_model_dir+'/selector.sav'
#         pickle.dump(selector, open(filename, 'wb'))
    
    
#     pd.DataFrame(X_val_tot).to_csv(save_model_dir+'/X_val_tot.csv')
#     if model != 'LR':
#         pd.DataFrame(shap_values_tot).to_csv(save_model_dir+'/shap_values_tot.csv')

#     pd.DataFrame(X_n_full).to_csv(save_model_dir+'/X_n_full.csv')
#     pd.DataFrame(Y).to_csv(save_model_dir+'/Y.csv')
    

# # ---------- NEWS ---------------------
# precision, recall,fpr,thresholds = results_news(X_n_full,y_real_full,'threshold')
# lab = 'Overall AP=%.4f' % (np.round(AP_manually(precision, recall),3))
# axes[0,0].step(recall, precision, label=lab, lw=2, color='black')


# axes[0,0].set_xlabel('Recall')
# axes[0,0].set_ylabel('Precision')
# axes[0,0].legend(loc='upper right', fontsize='small')
# axes[0,0].set_title('PR-curve NEWS')


# auc = metrics.auc(fpr, recall)
# lab = 'Overall AUC=%.4f' % (np.round(auc,3))
# axes[0,1].step(fpr, recall, label=lab, lw=2, color='black')

# axes[0,1].set_xlabel('FPR')
# axes[0,1].set_ylabel('Recall')
# axes[0,1].legend(loc='lower right', fontsize='small')
# axes[0,1].set_title('ROC NEWS')



# print('ap NEWS:', aps_NEWS,' \n mean:',np.mean(aps_NEWS), ' \n std:', np.std(aps_NEWS))
# print('ap Model:', aps_model,' \n mean:',np.mean(aps_model), ' \n std:', np.std(aps_model))
# print('auc NEWS:', aucs_NEWS,' \n mean:',np.mean(aucs_NEWS), ' \n std:', np.std(aucs_NEWS))
# print('auc Model:', aucs_model,' \n mean:',np.mean(aucs_model), ' \n std:', np.std(aucs_model))

    

# # ------------- Model ----------------------
# precision, recall, thresholds = precision_recall_curve(y_real_full, y_proba_full)
# lab = 'Overall AP=%.4f' % np.round(average_precision_score(y_real_full, y_proba_full),3)
# axes[1,0].step(recall, precision, label=lab, lw=2, color='black')


# axes[1,0].set_xlabel('Recall')
# axes[1,0].set_ylabel('Precision')
# axes[1,0].legend(loc='upper right', fontsize='small')
# axes[1,0].set_title('PR-curve Model')

# fpr, tpr, _ = metrics.roc_curve(y_real_full, y_proba_full)
# auc = metrics.auc(fpr, tpr)
# lab = 'Overall AUC=%.4f' % np.round(auc,3)
# axes[1,1].step(fpr, tpr, label=lab, lw=2, color='black')

# axes[1,1].set_xlabel('FPR')
# axes[1,1].set_ylabel('Recall')
# axes[1,1].legend(loc='lower right', fontsize='small')
# axes[1,1].set_title('ROC Model')


# f.tight_layout()
# if save_model:
#     f.savefig(save_results_dir+'/result.png',dpi=300)


# print('----- best fold random seed:',random_state_best)
# print(total_features)


# if model != 'LR':
#     import shap
#     from matplotlib import cm
#     from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#     newcmp = plt.get_cmap('cool')
#     plt.figure()
    
#     shap.summary_plot(shap_values_tot, features=X_val_tot, feature_names=total_features,plot_type='dot',show=False,max_display=25)
#     for fc in plt.gcf().get_children():
#         for fcc in fc.get_children():
#             if hasattr(fcc, "set_cmap"):
#                 fcc.set_cmap(newcmp)
#     plt.tight_layout()
#     plt.savefig(save_results_dir+'/Shap_summary_violin_full.png',dpi=200)
    
    
#     plt.figure()
#     shap.summary_plot(shap_values_tot, features=X_val_tot, feature_names=total_features,plot_type='bar',show=False,max_display=15)
#     plt.tight_layout()
#     plt.savefig(save_results_dir+'/Shap_summary_bar.png',dpi=200)
    
#     plt.figure()
#     sorted_idx = clf_best.feature_importances_.argsort()
#     plt.barh(np.asarray(total_features)[sorted_idx][-15:], clf.feature_importances_[sorted_idx][-15:])
#     plt.xlabel("Random Forest Feature Importance")
#     plt.tight_layout()
#     plt.savefig(save_results_dir+'/Gini_importance.png',dpi=200)
    
