import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from functions import *

class Parchure:
    def __init__(self,inputs,encoders,df):
        self.df = df                            # raw df with data
        self.df_train = pd.DataFrame            # df for train set
        self.df_val = pd.DataFrame              # df for validation set
        self.df_test = pd.DataFrame             # df for test set
        self.df_demo_train = pd.DataFrame       # df with demograhics for train set
        self.df_demo_val = pd.DataFrame         # df with demograhics for validation set
        self.df_demo_test = pd.DataFrame        # df with demograhics for test set
        self.X_train = []                       # Matrix with feature vectors for training set
        self.X_train_bal = []                   # Balanced matrix with feature vectors for training set
        self.y_train = []                       # label vector for train set
        self.y_train_bal = []                   # balanced label vector for train set
        self.X_val = []                         # Matrix with feature vectors for validation set
        self.y_val = []                         # label vector for validation set
        self.X_test = []                        # Matrix with feature vectors for test set
        self.y_test = []                        #label vector for test set
        self.features = []                      # array with names of variables   
        self.clf = None                         # model object
       


    def Prepare(self,random_state,val_share=0.2,test_share=0.2):
        
        self.df_train,self.df_val,self.df_test, self.df_demo_train,self.df_demo_val,self.df_demo_test = df_preparer(self.df,self.features,
                                                                            val_share,test_share,random_state) 
        
    
    def Build_feature_vectors(self,i,pred_window,gap,int_neg,int_pos,feature_window,name,label_type='mortality'):
       
                   
        print('TRAINING DATA')
        self.X_train,self.y_train,imputation_train,feature_imputation,_ = prepare_feature_vectors(self.df_train, self.df_train, self.df_demo_train,self.df_demo_train,
                                                                          self.ids_events,pred_window,gap,int_neg,int_pos,feature_window,self.features,
                                                                        label_type=label_type)
       
        
        print('VALIDATION DATA')
        self.X_val,self.y_val,imputation_val,_,self.val_pos = prepare_feature_vectors(self.df_val, self.df_train, self.df_demo_val,self.df_demo_train,
                                                                        self.ids_events,pred_window,gap,int_neg,int_pos,feature_window,self.features,
                                                                        label_type=label_type)
        print('TEST DATA')
        self.X_test,self.y_test,imputation_test,_,_ = prepare_feature_vectors(self.df_test, self.df_train, self.df_demo_test,self.df_demo_train,
                                                                        self.ids_events,pred_window,gap,int_neg,int_pos,feature_window,self.features,
                                                                        label_type=label_type)
    
    def Balance(self, undersampling = True):
        
        self.X_train_bal,self.y_train_bal = balancer(self.X_train,self.y_train,undersampling = undersampling)        
        
    def Train(self,model='RF',balance=True):
        if balance:
            
            self.clf,train_auc,self.explainer = train_model(self.X_train_bal,self.y_train_bal,self.X_test,self.y_test,model)
        else:
            self.clf,train_auc,self.explainer = train_model(self.X_train,self.y_train,self.X_test,self.y_test,model)
        
        return train_auc
        
    def Evaluate(self):
        auc,tn, fp, fn, tp,precision,recall = evaluate_metrics(self.clf,self.X_val,self.y_val)
        plot_roc_curve(self.clf, self.X_val, self.y_val)
        plot_PR_curve(precision,recall)
        return auc,tn, fp, fn, tp
