# -*- coding: utf-8 -*-
"""
Spyder Editor

Main script to run the ICU_model class


"""
print('main triggered')


# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import random

#import class and functions
from functions import *
from classes import *

# Define lists with input files
inputs = ['../../Data_EMC/20210208_covid data_lab.xlsx','../../Data_EMC/20210208_covid data_metingen.xlsx','../../Data_EMC/pirate.xlsx',
          '../../Data_EMC/MCOLS aanlevering _ 20210126_V3.xlsx']

# list with pd.read_excel encoders
encoders = ['utf-8',"ISO-8859-1",'ascii']


# Model characteristics
pred_window = 24    # [hours]
gap = 0             # [hours]
feature_window = 1  #[n samples back]
moving_feature_window = 1000    # [hours]
model = 'LR'        # type of classification model [RF,LR,SVM]
n_trees = 1000      # in case of RF, number of trees


# Data driven Feature selection (optional) --> select_K_best
FS = False 
n_keep = 50 # number of features to keep


# Imputation strategy
imputer='BR'    # imputer type: [KNN, BR (Bayesian Ridge)]
knn = 2 #in case inputer is KNN, n neighbours
initial_strategy = 'median' # in case imputer is BR (Bayesian Ridge)


# print results
prints_to_text = True
save_model= True
save_results_dir = '../results_17_02/no_LOS/SPO2_RR_only/int_val'
save_model_dir = '../results_17_02/no_LOS/SPO2_RR_only/int_val'
d = '../results_17_02/no_LOS/SPO2_RR_only/full_model' # dir to save fully-trained model results


k = 3 # k repititions of cross-validation

# Input characteristics
n_demo = ['BMI'] #demographics to include [AGE,SEX,BMI]

#sampling intervals
int_neg = 24     #negative patients
int_pos = 24    # positive patients

# CV shares
val_share = 0.4     # data share to be in validation set
test_share = 0.2    # data share in training set to be in test set (nested split)

# Make entry density plots
entry_dens = False
entry_dens_window = None    # if True, window to check entry density [hours]

balance=False
policy = True # To filter no-ICU policy patients
time=False   # To include LOS as a feature
NIV = False     # To inlcude Non-nvasive Ventilation features

# 'trend' Features
stats = True    # To include 'statistics' for vitals (max, min, mean, median, std, diff_std, diff_diff)
sliding_window = 24 # if stats = true, sliding window to collect stats features from 
diff = True     # include diff --> most recent value - 2nd most recent value (only vitals)
    

# Phycisian Behaviour features
freq = False     # frequency of requests [n observations / LOS]
inter = False   # time interval between most recent and 2nd most recent request
clocktime = False   # clocktime of request as a feature
    

# Early warning score 
NEWS = 'MEWS'       # type of Conventional EWS to be tested (NEWS / MEWS)


# make dict with model specs
specs = dict({'pred_window':pred_window,'gap':gap,'feature_window':feature_window,
              'int_neg':int_neg,'int_pos':int_pos,'val_share':val_share,'test_share':test_share,
              'model':model,'policy':policy,'freq':freq,'time':time,
              'inter':inter,'stats':stats,'diff':diff,'sliding_window':sliding_window,'n_demo':n_demo,
              'balance':balance,'knn':knn,'n_trees':n_trees,'FS':FS,
             'save_results':save_results_dir,'entry_dens':entry_dens,'moving_feature_window':moving_feature_window,
             'NIV':NIV,'imputer':imputer,'initial_strategy':initial_strategy,'clocktime':clocktime,'entry_dens_window':entry_dens_window,
             'NEWS':NEWS})




print('RESULTS FOR MODEL:',model)
print('PRED W:',pred_window,'Hours')
print('FEATURE W:',feature_window, 'samples')

#%%
# ICU_model = Class()
# data,data_vitals = ICU_model.import_MAASSTAD(inputs,encoders,specs)
# features = ICU_model.clean_MAASSTAD(specs)
# df_full,ids_events,df_demo = ICU_model.fix_episodes()
# X_MSD,dens_2,Y_MSD,X_MSD_full,ts = ICU_model.Build_feature_vectors(1,str(model)+'_'+str(n_features))
#%%

ICU_model = Class()
ICU_model.import_cci(inputs)
data, dict_unit = ICU_model.import_labs(inputs,encoders,specs)  # EMC data
unique_labs = ICU_model.clean_labs(specs)
ICU_model.import_vitals(inputs,encoders)
df_vitals = ICU_model.clean_vitals()
df_raw,features = ICU_model.merge()    #feature selection in this func
df_full, ids_events,df_demo = ICU_model.fix_episodes()
X_EMC,dens_2,Y_EMC,X_EMC_full,ts = ICU_model.Build_feature_vectors()



# %% TRAIN FULL MODEL

scaler,imputer,imputer_raw,clf,explainer,auc = ICU_model.Train_full_model(model,n_trees)  # need to save: model, imputer algo, standard scaler, shap explainer, performance (AUC)




import pickle
filename = d+'/trained_model.sav'
pickle.dump(clf, open(filename, 'wb'))

filename = d+'/scaler_best.sav'
pickle.dump(scaler, open(filename, 'wb'))

filename = d+'/explainer_best.sav'
pickle.dump(explainer, open(filename, 'wb'))

filename = d+'/imputer_best.sav'
pickle.dump(imputer, open(filename, 'wb'))

filename = d+'/imputer_raw.sav'
pickle.dump(imputer_raw, open(filename, 'wb'))






#%% Utility analysis
# from functions import *
# # Y = pd.read_csv('../COVID_PREDICT/Y_covid_predict.csv')
# # prev = 0.017016317016317017
# plot_Utility(Y_MSD,specs)



#%%  TO RUN INTERNAL CV


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# initiate figure for results
f, axes = plt.subplots(3, 2, figsize=(10, 10))

# make empty lists for metrics to keep track of
y_real = [] # labels
y_proba = [] # model probabilities
X_n = [] # dataframes needed to calculate NEWS
y_ts = []  # times between prediction and time of event (ICU trasfer or Discharge)
y_pats = [] # label neg/pos patient 

aps_NEWS = [] # avergae precisions EWS
aps_model = []# avergae precisions ML model  
aucs_NEWS = [] # AUCs EWS
aucs_model = [] # AUCs ML model
aps_tr = [] # apperent Average precision (on training set)
aucs_tr = [] # apperent AUCs (on training set)


for i in range(k):
    
    print('------ Fold',i,' -----')
    
    # Prepare the raw dataset: Data splits, normalization, imputation
    imputer_raw,imputer,y_pat,y_t,random_state,scaler = ICU_model.Prepare(random.randint(0, 10000))
    y_ts.append(y_t)
    y_pats.append(y_pat)
    
    # Make total feature vector in first fold
    if i == 0:
        total_features = ICU_model.total_feature_vector()
        print('Running model with features: \n ',total_features)
    
    # Optimize model using train and test set    
    clf,explainer,pred_tr,y_tr = ICU_model.Optimize() # Return trained model, SHAP explainer, Apparent performance (on train set)
    
    # Do predictions on validation set (internal validation), return label, probabilities, raw dataframe (for EWS) and normalized dataframe
    y_true,y_pred,X,X_val = ICU_model.Predict()
    y_real.append(y_true)
    y_proba.append(y_pred)
    X_n.append(X)
    
    # Get performance of EWS
    precision_n, recall_n,fpr_n,_= results_news(X,y_true,'threshold')
    auc_n = metrics.auc(fpr_n, recall_n)
    ap_n = AP_manually(precision_n, recall_n)
    aps_NEWS.append(ap_n)
    aucs_NEWS.append(auc_n)
    
    # Plot
    lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap_n,3))
    axes[0,0].step(recall_n, precision_n, label=lab)
    lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc_n,3))
    axes[0,1].step(fpr_n, recall_n, label=lab)

    # Apperent performance (TRAIN SET)
    precision, recall, _ = precision_recall_curve(y_tr, pred_tr)
    fpr, tpr, _ = metrics.roc_curve(y_tr, pred_tr)
    auc = metrics.auc(fpr, tpr)
    ap = average_precision_score(y_tr, pred_tr)
    aps_tr.append(ap)
    aucs_tr.append(auc)
    
    #Plot
    lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap,3))
    axes[1,0].step(recall, precision, label=lab)
    lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc,3))
    axes[1,1].step(fpr, tpr, label=lab)
    
    # performance on validation set
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    ap = average_precision_score(y_true, y_pred)
    aps_model.append(ap)
    aucs_model.append(auc)
    
    # Plot
    lab = 'Fold %d AP=%.4f' % (i+1, np.round(ap,3))
    axes[2,0].step(recall, precision, label=lab)
    lab = 'Fold %d AUC=%.4f' % (i+1, np.round(auc,3))
    axes[2,1].step(fpr, tpr, label=lab)
    
    
    # make total SHAP matrix (If using Random Forest)
    if i ==0:
        if (model != 'LR') and (model != 'NB'):
            shap_values = explainer.shap_values(X_val)
            shap_values_tot = shap_values[1]
        X_val_tot = X_val
    else:
        X_val_tot = np.concatenate([X_val_tot,X_val],axis=0)
        if (model != 'LR') and (model != 'NB'):
            shap_values_tot = np.concatenate([shap_values_tot,explainer.shap_values(X_val)[1]],axis=0)
    
    
    # ------ Update best fold --------
    if (i == 0) or (auc>auc_best):
        auc_best = auc
        clf_best = clf
        scaler_best = scaler
        explainer_best = explainer
        random_state_best = random_state
        imputer_best = imputer
        imputer_raw_best = imputer_raw
        
        print('Updated best AUC:',auc_best, ' for validation fold ', i)
        
        
print('best AUC found for validation fold:', i,' with: ', auc_best)


# concatenate collected lists
X_n_full = np.concatenate(X_n)
y_real_full = np.concatenate(y_real)
y_proba_full = np.concatenate(y_proba)
y_t_full = np.concatenate(y_ts)
y_pat_full = np.concatenate(y_pats)

# Make overview dataframe of labels, predictions, time-to-events and patient labels
Y = pd.DataFrame()
Y['label'] = y_real_full
Y['pred'] = y_proba_full
Y['time_to_event'] = y_t_full
Y['patient'] = y_pat_full

if prints_to_text: # print coming results in text file
    import sys
    sys.stdout = open(save_results_dir+'/results_'+str(pred_window)+'.txt','wt')



print('ap NEWS:', aps_NEWS,' \n mean:',np.mean(aps_NEWS), ' \n std:', np.std(aps_NEWS))
print('ap Model:', aps_model,' \n mean:',np.mean(aps_model), ' \n std:', np.std(aps_model))
print('ap TRAIN:', aps_tr,' \n mean:',np.mean(aps_tr), ' \n std:', np.std(aps_tr))
print('auc NEWS:', aucs_NEWS,' \n mean:',np.mean(aucs_NEWS), ' \n std:', np.std(aucs_NEWS))
print('auc Model:', aucs_model,' \n mean:',np.mean(aucs_model), ' \n std:', np.std(aucs_model))
print('auc TRAIN:', aucs_tr,' \n mean:',np.mean(aucs_tr), ' \n std:', np.std(aucs_tr))
print('----- best fold random seed:',random_state_best)
print(total_features)



# --- Save model and attributes of best fold for later analysis---
if save_model:
    import pickle
    filename = save_model_dir+'/trained_model.sav'
    pickle.dump(clf_best, open(filename, 'wb'))
    
    filename = save_model_dir+'/scaler_best.sav'
    pickle.dump(scaler_best, open(filename, 'wb'))
    
    filename = save_model_dir+'/explainer_best.sav'
    pickle.dump(explainer_best, open(filename, 'wb'))
    
    filename = save_model_dir+'/imputer_best.sav'
    pickle.dump(imputer_best, open(filename, 'wb'))
    
    filename = save_model_dir+'/imputer_raw_best.sav'
    pickle.dump(imputer_raw_best, open(filename, 'wb'))
    
    if FS:
        filename = save_model_dir+'/selector.sav'
        pickle.dump(selector, open(filename, 'wb'))
    
    
    pd.DataFrame(X_val_tot).to_csv(save_model_dir+'/X_val_tot.csv')
    if (model != 'LR') and (model != 'NB'):
        pd.DataFrame(shap_values_tot).to_csv(save_model_dir+'/shap_values_tot.csv')

    pd.DataFrame(X_n_full).to_csv(save_model_dir+'/X_n_full.csv')
    pd.DataFrame(Y).to_csv(save_model_dir+'/Y.csv')
    

# ---------- Fix axes labels / legend / titels for results figure  ---------------------
axes[0,0].set_xlabel('Recall')
axes[0,0].set_ylabel('Precision')
axes[0,0].legend(loc='upper right', fontsize='small')
axes[0,0].set_title('PR-curve '+str(NEWS))

axes[0,1].set_xlabel('FPR')
axes[0,1].set_ylabel('Recall')
axes[0,1].legend(loc='lower right', fontsize='small')
axes[0,1].set_title('ROC '+str(NEWS))

axes[1,0].set_xlabel('Recall')
axes[1,0].set_ylabel('Precision')
axes[1,0].legend(loc='upper right', fontsize='small')
axes[1,0].set_title('Apparent PR-curve')

axes[1,1].set_xlabel('FPR')
axes[1,1].set_ylabel('Recall')
axes[1,1].legend(loc='lower right', fontsize='small')
axes[1,1].set_title('Apparent ROC')
    
axes[2,0].set_xlabel('Recall')
axes[2,0].set_ylabel('Precision')
axes[2,0].legend(loc='upper right', fontsize='small')
axes[2,0].set_title('PR-curve Validation set')

axes[2,1].set_xlabel('FPR')
axes[2,1].set_ylabel('Recall')
axes[2,1].legend(loc='lower right', fontsize='small')
axes[2,1].set_title('ROC Validation set')


f.tight_layout()
if save_model:
    f.savefig(save_results_dir+'/result.png',dpi=300)


# if running Random Forest, make SHAP figures
if (model != 'LR') and (model != 'NB'):
    import shap
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    newcmp = plt.get_cmap('cool')
    plt.figure()
    
    shap.summary_plot(shap_values_tot, features=X_val_tot, feature_names=total_features,plot_type='dot',show=False,max_display=25)
    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(newcmp)
    plt.tight_layout()
    plt.savefig(save_results_dir+'/Shap_summary_violin_full.png',dpi=200)
    
    
    plt.figure()
    shap.summary_plot(shap_values_tot, features=X_val_tot, feature_names=total_features,plot_type='bar',show=False,max_display=15)
    plt.tight_layout()
    plt.savefig(save_results_dir+'/Shap_summary_bar.png',dpi=200)
    
    plt.figure()
    sorted_idx = clf_best.feature_importances_.argsort()
    plt.barh(np.asarray(total_features)[sorted_idx][-15:], clf_best.feature_importances_[sorted_idx][-15:])
    plt.xlabel("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(save_results_dir+'/Gini_importance.png',dpi=200)
    
#%% RE-RUN CLASS TO PREPARE MAASSTAD DATASET
ICU_model = Class()
data,data_vitals = ICU_model.import_MAASSTAD(inputs,encoders,specs)
features = ICU_model.clean_MAASSTAD(specs)
df_full,ids_events,df_demo = ICU_model.fix_episodes()
X_MSD,dens_2,Y_MSD,X_MSD_full,ts = ICU_model.Build_feature_vectors()



# %% LOAD MODEL
import pickle


filename = save_model_dir+'/trained_model.sav'
clf = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/scaler_best.sav'
scaler = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/explainer_best.sav'
explainer = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/imputer_best.sav'
imputer = pickle.load(open(filename, 'rb'))
filename = save_model_dir+'/imputer_raw_best.sav'
imputer_raw = pickle.load(open(filename, 'rb'))


# CHEK PERFORMANCE OF TRAINED MODEL ON MAASSTAD (EXTERNAL VALIDATION)
X = X_MSD
# X = np.delete(X,1,1)
y = Y_MSD.label


# calculate EWS
X_news,_ = build_news(X,X,specs)
precision_n, recall_n,fpr_n,_= results_news(X_news,y,'threshold')
auc_n = metrics.auc(fpr_n, recall_n)
ap_n = AP_manually(precision_n, recall_n)



y_pred,X,auc,ap = Predict_full_model(clf,scaler,imputer,X,y)    
from sklearn import datasets, metrics
metrics.plot_roc_curve(clf, X, y)
plt.step(fpr_n, recall_n, label='EWS')
plt.legend(['Model AUC='+str(np.round(auc,2)),'EWS AUC='+str(np.round(auc_n,2))])
plt.savefig(d + '/ROC_curve.png',dpi=300)

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y, y_pred)
plt.figure()
plt.plot(recall,precision,label='Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.step(recall_n, precision_n, label='EWS')
plt.legend(['Model AP=' + str(np.round(ap,2)) ,'EWS AP='+str(np.round(ap_n,2))])
plt.savefig(d + '/PR_curve.png',dpi=300)




#%%
# # in case of LR, plot coefs

if model == 'LR':
    objects = make_total_features(features,specs)
    y_pos = np.arange(len(objects))
    performance = clf.coef_[0,:]
    plt.figure()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.tick_params(axis="x", rotation=90)
    plt.ylabel('LR coeff')
    plt.title('Logistic regression')
    plt.tight_layout()
    plt.savefig(d+'/LR_coefss.png',dpi=300)

# from sklearn.inspection import plot_partial_dependence, partial_dependence
# # plot the partial dependence
# plot_partial_dependence(clf, X, [0, (0, 1)])

# t = 0.2
# label = 'pos'

# if label == 'pos':
#     X = X_MSD_full[np.where(y==1)]
# else:
#     X = X_MSD_full[~np.where(y==0)]
    
# # Proof of concept plot
# ICU_model.Proof_of_concept(clf,scaler,explainer,imputer,imputer_raw,X,ids_events,ts,t,plot=True)
        






