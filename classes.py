# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:43:59 2020

@author: 31643
"""
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from functions import *


class Parchure:
    def __init__(self,specs):
        self.specs = specs
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
    
    def import_cci(self,inputs):
        self.df_cci = importer_cci(inputs[2])
    def import_pacmed(self,features):
        self.features = features
        self.df,self.df_episodes = importer_pacmed(self.features)        
        
    def import_labs(self,inputs,encoders):
        self.df_lab,self.dict_unit = importer_labs(inputs[0],encoders[1],';',0,self.specs,filter=True)
        
        return self.df_lab, self.dict_unit
    def clean_labs(self):
        self.df_lab = cleaner_labs(self.df_lab)
        
    def import_vitals(self,inputs,encoders):
        self.df_vitals = importer_vitals(inputs[1],encoders[1],';',0)
        
    def clean_vitals(self):
        self.df_vitals = cleaner_vitals(self.df_vitals)
        
        
    def merge(self):
        
        self.df,self.features = df_merger(self.df_lab,self.df_vitals,self.df_cci,self.specs)
        self.ids_IC_only, self.ids_all, self.ids_clinic, self.ids_events = get_ids(self.df)
        
    
    def missing(self,x_days=True):
        df_missing,n_clinic,n_event = missing(self.df,self.features,self.ids_clinic,self.ids_events,x_days=x_days)
        
        return df_missing,n_clinic,n_event
    
    def Prepare_pacmed(self,random_state):
        self.df_train,self.df_val,self.df_test, self.df_demo_train,self.df_demo_val,self.df_demo_test = df_preparer_pacmed(self.df,self.df_episodes,
                                                                            random_state,self.specs) 
        self.df_train_raw,self.df_val_raw, self.df_test_raw, self.df_demo_train_raw,self.df_demo_val_raw, self.df_demo_test_raw = df_preparer_pacmed(self.df,self.df_episodes,
                                                                            random_state,self.specs,norm=False) 
        
        self.demo_median,self.median = Prepare_imputation_vectors(self.df_demo_train,self.df_train,self.features)
        self.demo_median_raw,self.median_raw = Prepare_imputation_vectors(self.df_demo_train_raw,self.df_train_raw,self.features)
        
    
    def Prepare(self,random_state):
        print('start preparing dataframes')
        self.df_train,self.df_val,self.df_test, self.df_demo_train,self.df_demo_val,self.df_demo_test = df_preparer(self.df,self.features,
                                                                            self.ids_IC_only,self.ids_events,
                                                                            random_state,self.specs) 
        self.df_train_raw,self.df_val_raw, self.df_test_raw , self.df_demo_train_raw,self.df_demo_val_raw, self.df_demo_test_raw = df_preparer(self.df,self.features,
                                                                   self.ids_IC_only,self.ids_events,
                                                                     random_state,self.specs,norm=False) 
        
        self.demo_median,self.median = Prepare_imputation_vectors(self.df_demo_train,self.df_train,self.features)
        self.demo_median_raw,self.median_raw = Prepare_imputation_vectors(self.df_demo_train_raw,self.df_train_raw,self.features)
        
        
        
        
    def Build_feature_vectors(self,i,name):
        print('TRAINING DATA')
                   

        self.X_train,self.y_train,imputation_train,feature_imputation,_,_ = prepare_feature_vectors(self.df_train, self.median , self.df_demo_train,self.demo_median,self.df_episodes,
                                                                          self.ids_events,self.features,self.specs)
        
        self.X_train_raw,_,_,_,_,_ = prepare_feature_vectors(self.df_train_raw, self.median_raw, self.df_demo_train_raw,self.demo_median_raw,self.df_episodes,
                                                                          self.ids_events,self.features,self.specs)
       
        
        print('VALIDATION DATA')
        self.X_val,self.y_val,imputation_val,_,self.val_pos,self.y_pat = prepare_feature_vectors(self.df_val, self.median , self.df_demo_val,self.demo_median,self.df_episodes,
                                                                        self.ids_events,self.features,self.specs)
        
        self.X_val_raw,_,_,_,_,_ = prepare_feature_vectors(self.df_val_raw, self.median_raw , self.df_demo_val_raw,self.demo_median_raw,self.df_episodes,
                                                                        self.ids_events,self.features,self.specs)
        
        print('TEST DATA')
        self.X_test,self.y_test,imputation_test,_,_,_ = prepare_feature_vectors(self.df_test, self.median , self.df_demo_test,self.demo_median,self.df_episodes,
                                                                        self.ids_events,self.features,self.specs)
        
        self.X_test_raw,_,_,_,_,_ = prepare_feature_vectors(self.df_test_raw, self.median_raw , self.df_demo_test_raw,self.demo_median_raw,self.df_episodes,
                                                                        self.ids_events,self.features,self.specs)
        
        
        if i == 1:
            general_imputation = pd.DataFrame([imputation_train,imputation_val,imputation_test])
            df = stacked_barchart(['train','val','test'],general_imputation,'general_imp_'+name)
            df = stacked_barchart(['BMI','AGE']+list(self.features),feature_imputation,'feature_sepc_imp_'+name)
            
        self.X_train,self.X_train_raw,self.X_val,self.X_val_raw,self.X_test,self.X_test_raw,self.imputer,self.imputer_raw = KNN_imputer(self.X_train,self.X_train_raw,
                                                                                                          self.X_val,self.X_val_raw,
                                                                                                          self.X_test,self.X_test_raw,
                                                                                                          self.specs)
        
        return self.imputer,self.imputer_raw
    
    def Balance(self, undersampling = True):
        
        self.X_train_bal,self.y_train_bal = balancer(self.X_train,self.y_train,undersampling = undersampling)
        # self.X_val,self.y_val = balancer(self.X_val,self.y_val,undersampling = undersampling)
        # self.X_test,self.y_test = balancer(self.X_test,self.y_test,undersampling = undersampling)
        
        
    def Train(self,w):
        if self.specs['balance']:
            
            self.clf,train_auc,self.explainer,_,_,_,t = train_model(self.X_train_bal,self.y_train_bal,self.X_test,self.y_test,self.specs['model'])
        else:
            self.clf,train_auc,self.explainer,_,_,_,t = train_model(self.X_train,self.y_train,self.X_test,self.y_test,self.specs['model'],class_weight={0:1,1:w})
        
        return t
    
  

    def Optimize_trees(self):
        
        n_trees = optimize_n_trees(self.X_train,self.y_train)
        
    def feature_selection(self,n):
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
            print(self.total_features[-15:])
            
        return self.selector
        
        
        
    def Optimize_weights(self):
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
            10,15,20,25
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
        plt.savefig('results/PR_optimization',dpi=300)
        
        return t_best
    
    def Predict(self):
        
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
        print('Average precision on validation set:',ap)
        # NEWS
        X_train, X_val = build_news(self.X_train_raw,self.X_val_raw,1) 
        precision_n, recall_n,fpr_n,_= results_news(X_val,self.y_val,'threshold')
        auc_n = metrics.auc(fpr_n, recall_n)
        ap_n = AP_manually(precision_n, recall_n)
        
        #Dummy
        model = DummyClassifier(strategy='stratified')
        model.fit(self.X_train, self.y_train)
        yhat = model.predict_proba(self.X_val)
        naive_probs = yhat[:, 1]
        
        return self.clf,self.explainer,self.df_val,self.median,self.df_demo_val,self.demo_median,self.df_val_raw,self.median_raw,self.df_demo_val_raw,self.demo_median_raw,precision,recall,ap,fpr,tpr,auc,self.y_val,predictions,precision_n,recall_n,fpr_n,ap_n,auc_n,X_val,naive_probs,self.X_val
    
        
    def Evaluate(self,beta):

        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score
        
        # NEWS PERFORMANCE
        print('NEWS:')
        X_train, X_val = build_news(self.X_train_raw,self.X_val_raw,self.specs['feature_window'])
        precision_news, recall_news= results_news(X_val,self.y_val,'threshold')

        
        # MODEL PERFORMANCE
        print('\n MODEL')
        auc,ap,tn, fp, fn, tp,precision,recall,t = evaluate_metrics(self.clf,self.X_val,self.y_val,plot=True)
        
        # do classification with dummy classifier
        model = DummyClassifier(strategy='stratified')
        model.fit(self.X_train, self.y_train)
        yhat = model.predict_proba(self.X_val)
        naive_probs = yhat[:, 1]
        precision_naive, recall_naive, _ = precision_recall_curve(self.y_val, naive_probs)
        ap_naive = average_precision_score(self.y_val, naive_probs)
        print('Naive AP:',ap_naive)
    
        # MAKE PR and ROC curves
        plot_PR_curve(precision,recall,precision_news,recall_news,precision_naive, recall_naive)
        # plot_roc_curve(self.clf, self.X_val, self.y_val,clf, X_val, y_val)
        
        #Model PERFORMANCE ONLY NEG PATIENTS
        print('\n Perfromance among negatve patients')
        mask = self.y_pat == 0
        CM(self.clf,self.X_val[mask],self.y_val[mask],t)
        
        #Model PERFORMANCE ONLY POS PATIENTS
        print('\n Perfromance among positive patients')
        mask = self.y_pat == 1
        CM(self.clf,self.X_val[mask],self.y_val[mask],t)
        return auc,ap,tn, fp, fn, tp
    
        
        
    
    def Plot_results(self,n,model):
        # 
        FI = plot_feature_importance(self.clf,n,model)
        
        return FI,self.features
     
    def Global_Feature_importance(self,X_val,explainer,clf):
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
    
    def Proof_of_concept(self,clf,explainer,imputer,imputer_raw,
                         df_val,median,df_demo_val,demo_median,
                         df_val_raw,median_raw,df_demo_val_raw,demo_median_raw,
                         t,inc_start,label,plot=True):
        import shap
        import matplotlib.pyplot as plt
        
        shap.initjs()
        df = isolate_class(df_val,label)
        idxs = np.unique(df['ID'])
        
        
        

        for i in range(
                len(idxs)
                # 1
                ):
            print('\n Patient',i)
            # try:
            X,X_raw,ts,t_event = prepare_feature_vectors_plot(df,median,df_demo_val,demo_median,
                                                        df_val_raw,median_raw,df_demo_val_raw,demo_median_raw,
                                                        self.features,i,self.specs,inc_start=inc_start,label=label)
            from numpy import inf
            X[X == inf] = np.nan
            X[X == -inf] = np.nan
            X_raw[X_raw == inf] = np.nan
            X_raw[X_raw == -inf] = np.nan
            
            X = imputer.transform(X)
            X_raw_imputed = imputer_raw.transform(X_raw)
            
            print('shape X:',X.shape)
            print('shape X_raw:',X_raw.shape)
            print('shape df demo:',df_demo_val.shape)
            
            # transform X_raw into pd Dataframe
            
            features_tot = self.total_features
            X_raw = pd.DataFrame(X_raw,columns=features_tot)
            X_raw_imputed = pd.DataFrame(X_raw_imputed,columns=features_tot)
            
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
