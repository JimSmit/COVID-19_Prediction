# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:43:59 2020

@author: Jim Smit

ICU_model CLass

"""

# import libraries
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

#import functions
from functions import *


class Class:
    def __init__(self):

        self.df_lab = pd.DataFrame
        self.df_vitals = pd.DataFrame
        self.df_episodes = pd.DataFrame
        self.df = pd.DataFrame
        self.df_train = pd.DataFrame
        self.df_val = pd.DataFrame
        self.df_test = pd.DataFrame
        self.df_demo_train = pd.DataFrame
        self.df_demo_val = pd.DataFrame
        self.df_demo_test = pd.DataFrame
        self.ids_all = []
        self.ids_IC_only = []
        self.ids_clinic = []
        self.ids_events = []
        self.X_train = []
        self.X_train_bal = []
        self.y_train = []
        self.y_train_bal = []
        self.X_val = []
        self.val_pos = []
        self.y_val = []
        self.X_test = []
        self.y_test = []
        self.demo_median = []
        self.median = []
        self.features = []
        self.clf = None
        self.explainer = None
        self.df_cci = pd.DataFrame
        self.clf_NEWS = None
    
    def import_cci(self,inputs):  # file with No ICU policy information
        self.df_cci = importer_cci(inputs[2])
       
    def import_MAASSTAD(self,inputs,encoders,specs): # Load Maasstad file
        self.specs = specs
        self.df_imported,df_vitals = importer_MAASSTAD(inputs[3],encoders[1],',',0,self.specs)
        return self.df_imported,df_vitals
    
    def clean_MAASSTAD(self,specs): # Clean Maasstad file
        self.specs = specs
        self.df_cleaned,self.features = cleaner_MAASSTAD(self.df_imported,self.specs)
        # self.ids_IC_only, self.ids_all, self.ids_clinic, self.ids_events = get_ids(self.df_cleaned)
        
        return self.features
    
    def import_labs(self,inputs,encoders,specs): #load Labs EMC
        self.specs = specs
        self.df_lab_raw,self.dict_unit = importer_labs(inputs[0],encoders[1],';',0,self.specs,filter=True)
        
        return self.df_lab, self.dict_unit
    
    def clean_labs(self,specs):  # Clean Labs EMC
        self.specs = specs
        self.df_lab,unique_features = cleaner_labs(self.df_lab_raw)
        return unique_features
    
    def import_vitals(self,inputs,encoders): # Import vitals EMC
        self.df_vitals_raw = importer_vitals(inputs[1],encoders[1],';',0)
        
    def clean_vitals(self): # clean Vitals EMC
        self.df_vitals = cleaner_vitals(self.df_vitals_raw)
        return self.df_vitals
        
    def merge(self): # Merge labs and vitals EMC
        
        self.df_cleaned,self.features = df_merger(self.df_lab,self.df_vitals,self.df_cci,self.specs)
       
        return self.df_cleaned,self.features
         
    
    def fix_episodes(self): # Go from patients to patient episodes (to handle re-admissions)
        # self.df,self.ids_events = fix_episodes(self.df_cleaned)
        self.df,self.ids_events = fix_episodes(self.df_cleaned,self.specs)
        self.df_demo = Demographics(self.df,self.specs)
        
        return self.df,self.ids_events,self.df_demo
    
    def Build_feature_vectors(self): # transform dataframe into matrix with feature vectors
        
        self.X,self.y,entry_dens,self.y_pat,self.y_t,y_entry_dens,ts = prepare_feature_vectors(self.df,self.df_demo,
                                                                                   self.ids_events,self.features,self.specs)
        
        # get rid of infinity in X (if present)
        from numpy import inf
        self.X[np.where(self.X == np.inf)] = np.nan
        
        # transform all to floats
        X_return  = self.X[:,:-1].astype(float)
        
        
        if self.specs['entry_dens']:
            print('PLOT ENTRY DENSITIES')
            
            entry_dens = pd.DataFrame(entry_dens)
            print(entry_dens.shape)
            
            print(make_total_features(self.features,self.specs,demo=False))
            entry_dens.columns = make_total_features(self.features,self.specs,demo=False)
            
            plot_df = pd.DataFrame()
            
            entry_dens = entry_dens.iloc[:,:5]
            
            for i in entry_dens.columns:
                df = pd.DataFrame()
                df['entry density'] = entry_dens[i]
                df['Variable'] = i
                df['Label'] = 'No transfer'
                df['Label'].loc[y_entry_dens==1] = 'Transfer'
                plot_df = pd.concat([plot_df,df],axis=0)
            
            plot_df.columns = ['entry density','Variable','Label'] 
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure()
            ax = sns.boxplot(x="Variable", y="entry density", hue = 'Label',data=plot_df,medianprops={'color':'red'})
            plt.ylim(0,1)
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            plt.savefig('entry_vitals_MSD',dpi=300)
        
        # Prapere Y with label characteristics (for Utility aa=nalysis)
        Y = pd.DataFrame()
        Y['label'] = self.y
        Y['t_to_event'] = self.y_t
        Y['patient'] = self.y_pat
        Y['ID'] = self.X[:,-1]
        
        
        return X_return,entry_dens,Y,self.X,ts
    
    
    def Train_full_model(self,model,n_trees): # Train model on full dataset
        # Train imputer for raw data
        print('start iputation with:',self.specs['imputer'])
        _,self.imputer_raw = Imputer_full_model(self.X[:,:-1],self.specs) 
        # Normalize
        self.X_norm,self.scaler = Normalize_full(self.X[:,:-1]) 
        #Imputation
        print('start iputation with:',self.specs['imputer'])
        self.X_norm_imp,self.imputer = Imputer_full_model(self.X_norm,self.specs) 
        #Train
        self.clf,self.explainer,auc = train_full_model(self.X_norm_imp,self.y,model,n_trees)

        return self.scaler,self.imputer,self.imputer_raw,self.clf,self.explainer,auc
            
    def Prepare(self,random_state): # Data split, normalization and imputation for internal validation
        print('start preparing dataframes')
        
        self.X_train_raw,self.y_train,self.X_val_raw,self.y_val,self.y_val_pat,self.y_val_t,self.X_test_raw,self.y_test = Split(self.X,
                                                                                                self.y,self.y_pat,self.y_t,self.ids_events,
                                                                                                          random_state,self.specs) 
        
        # Normalize
        self.X_train,self.X_val,self.X_test,self.scaler = Normalize(self.X_train_raw,self.X_val_raw, self.X_test_raw,self.specs) 
        
        #Imputation
        print('start iputation with:',self.specs['imputer'])
        self.X_train_raw,self.X_val_raw,self.X_test_raw,self.imputer_raw = Imputer(self.X_train_raw,
                                                                                           self.X_val_raw, self.X_test_raw,self.specs) 
        
        self.X_train,self.X_val,self.X_test,self.imputer = Imputer(self.X_train,self.X_val,self.X_test,self.specs) 
        
        
        return self.imputer_raw,self.imputer,self.y_val_pat,self.y_val_t,random_state,self.scaler
    
  
    def total_feature_vector(self): # build total feature vector (feature names)
        print('make total feature vector')
        self.total_features = make_total_features(self.features,self.specs)
        print(self.X_train.shape)
        print(self.total_features.shape)
        
        return list(self.total_features)
        
    def feature_selection(self,n): # Optional: data driven feature selection
        print('feature selection triggered')
        self.total_features = make_total_features(self.features,self.specs)
        print(self.X_train.shape)
        print(self.total_features.shape)
        
        top_idx = 0
        self.selector = None
        
        if self.specs['FS']:
            print('Perform Feature selection, keep only top',n)
            from sklearn.feature_selection import SelectKBest, chi2, f_classif
            self.selector = SelectKBest(f_classif, k=n).fit(self.X_train, self.y_train)
        
            mask = self.selector.get_support() #list of booleans
            new_features = [] # The list of your K best features
            for bool, feature in zip(mask, self.features):
                if bool:
                    new_features.append(feature)
            
            
            self.total_features = np.asarray(new_features)
            print(self.total_features)
            
        return self.selector,list(self.total_features)
        
    def Optimize(self): #optimize model with standard balanced weights
        print('start optimizing with standard balanced weights')
        if self.specs['FS']:
            self.X_train = self.selector.transform(self.X_train)
            self.X_val = self.selector.transform(self.X_val)
            self.X_test = self.selector.transform(self.X_test)
        
        self.clf,self.explainer,_,_,pred_tr,y_tr = train_model(self.X_train,self.y_train,self.X_test,self.y_test,self.specs['model'],n_trees=self.specs['n_trees'],class_weight='balanced')
        
        return self.clf,self.explainer,pred_tr,y_tr
        
    def Optimize_weights(self): # Optional: optimize class weights with gridsearch
        print('start optimizing weights')
        import matplotlib.pyplot as plt
        
        
        if self.specs['FS']:
            self.X_train = self.selector.transform(self.X_train)
            self.X_val = self.selector.transform(self.X_val)
            self.X_test = self.selector.transform(self.X_test)
        
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([-0.05,1.05])
        ax.set_ylim([-0.05,1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR Curve')
        weights = [
            # 15        
            1,5,10,15,20,30
                   ]
        best_ap = 0
        t_best = 0        
        for w,k in zip(weights,'bgrcmykw'):
            print('working on class weight:',w)
  
            clf,train_auc,explainer,p,r,ap,t = train_model(self.X_train,self.y_train,self.X_test,self.y_test,self.specs['model'],n_trees=self.specs['n_trees'],class_weight={0:1,1:w})
                
            ax.plot(r,p,c=k,label=w)
            if ap > best_ap:
                best_ap = ap
                print('better AP with weight:',w)
                self.clf = clf
                self.explainer = explainer
                t_best = t
        ax.legend(loc='lower left')    

        
        return t_best
    
    def Predict(self): # Make predictions on validation set
        
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score
        from sklearn.dummy import DummyClassifier
        from sklearn import metrics
        
        # MODEL

        predictions = self.clf.predict_proba(self.X_val)[:,1]
            
        precision, recall, _ = precision_recall_curve(self.y_val, predictions)
        fpr, tpr, _ = metrics.roc_curve(self.y_val, predictions)
        auc = metrics.auc(fpr, tpr)
        ap = average_precision_score(self.y_val, predictions)
        print('AUC on validation set:',auc)
        print('Average precision on validation set:',ap)
        # NEWS
        X_train, X_val = build_news(self.X_train_raw,self.X_val_raw,self.specs) 
        precision_n, recall_n,fpr_n,_= results_news(X_val,self.y_val,'threshold')
        auc_n = metrics.auc(fpr_n, recall_n)
        ap_n = AP_manually(precision_n, recall_n)
        
        
        print('N feature vectors in validation fold:',len(self.y_val))
        print('Prevalence in validation fold:',sum(self.y_val)/len(self.y_val))
        
        return self.y_val,predictions,X_val,self.X_val
    
         
    def Global_Feature_importance(self,X_val,explainer,clf): # make polots for global feature importance
        import shap
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        # Create colormap
        
        # Define colormap
        newcmp = plt.get_cmap('cool')
        
        
        
        # First only LOS
        features =  ['BMI',' AGE','LOS']
        LOS = X_val[:,:3]
        shap_values = explainer.shap_values(X_val)
        shap_values = shap_values[1]
        shap_values = shap_values[:,:3]
        plt.figure()
        shap.summary_plot(shap_values, features=LOS, feature_names=features,plot_type='dot',show=False)
        # Change the colormap of the artists
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if hasattr(fcc, "set_cmap"):
                    fcc.set_cmap(newcmp)
        
        plt.tight_layout()
        plt.savefig('results/Shap_summary_LOS.png',dpi=200)
              
        # # Without LOS
        # rest = np.concatenate([X_val[:,:2],X_val[:,3:]],axis=1)
        # print('shape rest:',rest.shape)
        # features =  ['BMI','AGE',]+list(self.total_features)
        # shap_values = explainer.shap_values(X_val)
        # shap_values = shap_values[1]
        # shap_values = np.concatenate([shap_values[:,:2],shap_values[:,3:]],axis=1)
        # plt.figure()
        # shap.summary_plot(shap_values, features=rest, feature_names=features,plot_type='dot',show=False,max_display=15)
        # for fc in plt.gcf().get_children():
        #     for fcc in fc.get_children():
        #         if hasattr(fcc, "set_cmap"):
        #             fcc.set_cmap(newcmp)
        # plt.tight_layout()
        # plt.savefig('results/Shap_summary_violin.png',dpi=200)
        
        
        
        # Full
        features =  list(self.total_features)
        shap_values = explainer.shap_values(X_val)
        shap_values = shap_values[1]
        plt.figure()
        shap.summary_plot(shap_values, features=X_val, feature_names=features,plot_type='dot',show=False,max_display=15)
        for fc in plt.gcf().get_children():
            for fcc in fc.get_children():
                if hasattr(fcc, "set_cmap"):
                    fcc.set_cmap(newcmp)
        plt.tight_layout()
        plt.savefig('results/Shap_summary_violin_full.png',dpi=200)
        
        
        plt.figure()
        shap.summary_plot(shap_values, features=X_val, feature_names=features,plot_type='bar',show=False,max_display=15)
        plt.tight_layout()
        plt.savefig('results/Shap_summary_bar.png',dpi=200)
        
        
        plt.figure()
        sorted_idx = clf.feature_importances_.argsort()
        plt.barh(np.asarray(features)[sorted_idx][-15:], clf.feature_importances_[sorted_idx][-15:])
        plt.xlabel("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig('results/Gini_importance.png',dpi=200)
        
        return 
    
    def Proof_of_concept(self,clf,scaler,explainer,imputer,imputer_raw,  # Make dynamic plots for local feature importance / predicions
                         X,ids_events,ts,t,plot=True):
        import shap
        import matplotlib.pyplot as plt
        shap.initjs()
        
        # all available indexes:
        
        idx = np.unique(X[:,-1])
        
        
        for i in range(
                len(idxs)
                # 1
                ):
            print('\n Patient',i)
            # try:
            
            patient = X[np.where(X[:,-1] == idx)][:,:-1]
            print(patient.shape)
            
            total_features = make_total_features(self.features,self.specs)
            print(total_features.shape)
            
            patient = pd.DataFrame(patient,columns=total_features)
            #HEREEE
            if label == 'pos':
                print('to ICU:',t_event)
            else:
                print('discharge:',t_event)
            
            # Calculate model risks, PLOT SHAPLEY FORCE PLOTS   
            predictions =  predict(clf, X)
            
            diff = []
            if len(predictions) > 2:
                for p in range(len(predictions)-1):
                    diff.append(np.abs(predictions[p+1]-predictions[p]))
                diff = np.asarray(diff)
                n = len(predictions)
                if label == 'pos':
                    diff_idx = diff.argsort()[-(n-1):]
                else:
                    diff_idx = diff.argsort()[-3:]
                
                
                feature_inc_units = []
                for feature in features_tot:
                    feature_inc_units.append(feature+' '+self.dict_unit[feature])
                feature_inc_units = np.asarray(feature_inc_units)
                    
                count = 1    
                if plot:
                    for idx in diff_idx:
                        # new_base_value = np.log(t / (1 - t))  # the logit function
                        shap_display = shap.force_plot(
                                            explainer.expected_value[1], 
                                            # new_base_value,
                                            # link='logit',
                                            explainer.shap_values(X[idx+1,:])[1], 
                                            features=np.round(X_raw.iloc[idx+1,:],2), 
                                            feature_names=feature_inc_units,
                                            text_rotation=30,
                                            matplotlib=True,show=False, 
                                            # plot_cmap=["#FF5733","#335BFF"]
                                            )
            
                        plt.savefig('results/POC_plot_FORCE_'+ str(i) + '_' + str(count) + '.png',bbox_inches='tight',dpi=300)
                        count+=1
                    
            #Calculate feature impacts
            feature_impacts = list()
            
            for j in range(X.shape[0]):
                feature_impacts.append(explainer.shap_values(X[j,:])[1])
                
            
            feature_impacts = np.array([np.array(x) for x in feature_impacts])
            feature_impacts = pd.DataFrame(feature_impacts)
            feature_impacts.columns = features_tot
    
            # Calculate NEWS score
            news = []
            for v in range(X_raw_imputed.shape[0]):
                a = NEWS(X_raw_imputed.loc[v,'SpO2'],
                         X_raw_imputed.loc[v,'HR'],
                         X_raw_imputed.loc[v,'BP'],
                         X_raw_imputed.loc[v,'RR'],
                         X_raw_imputed.loc[v,'Temp']
                         )
                news.append(a)
            
            # 'Global' SHAPs for specific patient
            shap_values = explainer.shap_values(X)
            shap_mean = np.mean(np.abs(shap_values[1]),axis=0)
            sorted_idx = shap_mean.argsort()
            sorted_idx = list(sorted_idx)
            sorted_idx.remove(0)
            sorted_idx.remove(1)
            sorted_idx = np.asarray(sorted_idx)
            
            features_to_plot = features_tot[sorted_idx][-8:]
            
            if plot:
                plt = subplot(X_raw,ts,predictions,news,features_to_plot,i,t_event,feature_impacts,label,t,self.dict_unit,self.specs)  
            
            if i == 0:
                X_overall = X
            else:
                X_overall = np.concatenate([X_overall,X],axis=0)
            # except:
            #     print('patient', i, ' too short')
        
        
            
        
        return X_overall
