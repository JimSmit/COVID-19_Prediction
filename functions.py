# Import necessary libraries
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import sys

def df_preparer(df,variables, val_share=0.25,test_share=0.2,random_state=0,norm=True):
    
    print('df_preparer triggered')
    
    """
    Prepares the train, test and validation dataframes. 
    Performs the splits and normalizes the data based on the train set.

    Parameters
    ----------
    df : pd.DataFrame
        The raw single-timestamp data for all patients. This df shoul contain the following columns: ['ID','BMI','AGE','DEST',
        'OPNAMETYPE','TIME','VAR','VALUE']
    variables: np.array[str]
        Array with strings representing the variable names to be included in the model (excluding demographics).
    val_share: Optional[float]
        Fraction of patients to be used in the validation set.
    val_share: Optional[float]
        Fraction of patients to be used in the test set.
    random_state: Optional[int]
        Random seed for the train-test/validation-splits. 
    norm: optional:[Bool]
        Flag to turn on normalization by standarziation. 
        
    Returns
    -------
    df_train,df_val,df_test,df_demo_train,df_demo_val,df_demo_test
    type : pd.DataFrame
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # create df with Demographics data (BMI and AGE) -- >  df_demo
    
    ids = np.unique(df['ID']) # Unique IDs in raw dataset
    bmi = []
    age =[]
    for i in ids:
        bmi.append(df[df['ID']==i].reset_index()['BMI'][0])
        age.append(df[df['ID']==i].reset_index()['AGE'][0])
    df_demo = pd.DataFrame()
    df_demo['ID'] = ids
    df_demo['BMI'] = bmi
    df_demo['AGE'] = age
    
    
    # Split raw df in training and validation set on patient level:
    ids_train,ids_val = train_test_split(ids, test_size=val_share,random_state=random_state)
    
    df_train_full = df[df['ID'].isin(ids_train)]
    df_val = df[df['ID'].isin(ids_val)]
    
    df_demo_train_full = df_demo[df_demo['ID'].isin(ids_train)]
    df_demo_val = df_demo[df_demo['ID'].isin(ids_val)]

    #split training set in training and testing set
    ids = np.unique(df_train_full['ID'])
    ids_train,ids_test = train_test_split(ids, test_size=test_share,random_state=random_state)
    
    df_train = df_train_full[df_train_full['ID'].isin(ids_train)]
    df_test = df_train_full[df_train_full['ID'].isin(ids_test)]

    df_demo_train = df_demo_train_full[df_demo_train_full['ID'].isin(ids_train)]
    df_demo_test = df_demo_train_full[df_demo_train_full['ID'].isin(ids_test)]
    
    
    
    print('Split train, val en test set: \n original shape: ',df.shape,
          '\n train shape: ',df_train.shape, 'unique patients: ', len(df_train['ID'].unique()),
          '\n Val shape: ',df_val.shape, 'unique patient: ', len(df_val['ID'].unique()),
          '\n Test shape: ',df_test.shape, 'unique patients: ', len(df_test['ID'].unique()))
    
    
    # Normaize data using standardization
    if norm:
                
        df_train_norm = pd.DataFrame() # intialize empty normalized train, val and test set.
        df_val_norm = pd.DataFrame()
        df_test_norm = pd.DataFrame()
        
        for v in variables: # loop trough unique variables
    
            train_idx = (df_train.VARIABLE == v) # Define indices in train set for this variable
            val_idx = (df_val.VARIABLE == v) # Define indices in validation set for this variable
            test_idx = (df_test.VARIABLE == v) # Define indices in test set for this variable
                    
            scaler = StandardScaler()
            scaler.fit(df_train.loc[train_idx,'VALUE'].values.reshape(-1, 1)) # Fit scaler only on training set
            
            temp = df_train.loc[train_idx,'VALUE'].copy() #define temporary copy of Values from training df from only this variable.
            if temp.shape[0] == 0:
                print(v,'not in training set')
            else:
                temp = scaler.transform(temp.values.reshape(-1, 1))
                snip = df_train.loc[train_idx]
                snip = snip.assign(VALUE=temp)
                df_train_norm = pd.concat([df_train_norm,snip],axis=0)
            
            temp = df_val.loc[val_idx,'VALUE'].copy()
            if temp.shape[0] == 0:
                print(v,'not in validation set')
            else:
                temp = scaler.transform(temp.values.reshape(-1, 1))
                snip = df_val.loc[val_idx]
                snip = snip.assign(VALUE=temp)
                df_val_norm = pd.concat([df_val_norm,snip],axis=0)
            
            temp = df_test.loc[test_idx,'VALUE'].copy()
            if temp.shape[0] == 0:
                print(v,'not in test set')
            else:
                temp = scaler.transform(temp.values.reshape(-1, 1))
                snip = df_test.loc[test_idx]
                snip = snip.assign(VALUE=temp)
                df_test_norm = pd.concat([df_test_norm,snip],axis=0)
           
        df_train =df_train_norm    
        df_val =df_val_norm 
        df_test =df_test_norm 
        
        # normalize demographics
        
        df_demo_train_norm = pd.DataFrame()
        df_demo_val_norm = pd.DataFrame()
        df_demo_test_norm = pd.DataFrame()
        
        df_demo_train_norm['ID'] = df_demo_train['ID']
        df_demo_val_norm['ID'] = df_demo_val['ID']
        df_demo_test_norm['ID'] = df_demo_test['ID']
        
        
        for col in ['AGE','BMI']:
            
            scaler = StandardScaler()
            scaler.fit(df_demo_train[col].values.reshape(-1, 1))
            
            temp = df_demo_train.loc[:,col].copy()
            temp = scaler.transform(temp.values.reshape(-1, 1))
            df_demo_train_norm[col] = temp
            
            temp = df_demo_val.loc[:,col].copy()
            temp = scaler.transform(temp.values.reshape(-1, 1))
            df_demo_val_norm[col] = temp
            
            temp = df_demo_test.loc[:,col].copy()
            temp = scaler.transform(temp.values.reshape(-1, 1))
            df_demo_test_norm[col] = temp
          
        df_demo_train = df_demo_train_norm
        df_demo_val = df_demo_val_norm
        df_demo_test = df_demo_test_norm
        
        print('data normalized using standardscaler')
    
    # Make sure dfs for demographics and other variables contain same amount of patients
    assert(len(np.unique(df_train['ID']))==len(np.unique(df_demo_train['ID'])))
    assert(len(np.unique(df_val['ID']))==len(np.unique(df_demo_val['ID'])))
    assert(len(np.unique(df_test['ID']))==len(np.unique(df_demo_test['ID'])))

    # Make sure patients IDs in test/validation set are not present in train set
    assert(any(i in np.unique(df_val['ID']) for i in np.unique(df_train['ID'])) == False)
    assert(any(i in np.unique(df_test['ID']) for i in np.unique(df_train['ID'])) == False)
    assert(any(i in np.unique(df_demo_val['ID']) for i in np.unique(df_demo_train['ID'])) == False)
    assert(any(i in np.unique(df_demo_test['ID']) for i in np.unique(df_demo_train['ID'])) == False)

    return df_train,df_val,df_test,df_demo_train,df_demo_val,df_demo_test
    
    

def prepare_feature_vectors(df,df_train,df_demo,df_demo_train,df_patients,ids_IC_only,ids_events,pred_window,gap,int_neg,int_pos,feature_window,
                        features,label_type='mortality'):
    print('prepare_feature_vectors triggered')

    """
    Samples feature vectors from the input dfs. 

    Parameters
    ----------
    df : pd.DataFrame
        df to sample from.
    df_train: pd.DataFrame
        df containing training set. Imputed values are based on the training set. 
    df_demo: pd.DataFrame
        demograhics df to sample from.
    df_demo_train: pd.DataFrame
        demograhics df containing the training set. Imputed values are based on the training set. 
    pred_window: int
        Size of prediction window in hours
    gap: int
        Size of gap in hours
    int_neg: int
        Interval between samples for the negative class. In hours.
    int_pos: int
        Interval between samples for the positive class. In hours.
    feature_window: int
        Number of most recent assessments to be included in feature vector
    variables: np.array[str]
        Array of strings representing the names of the variables to be included in the model.


    Returns
    -------
    X: matrix [N feature vectors x N variables]
    y: vector [N feature vectors x 1]
    type : np.array
    """

    from datetime import datetime, timedelta

    pos = list() #create empty list for pos labeled feature vectors
    neg = list() #create empty list for neg labeled feature vectors
    
    if label_type == 'mortality':
        print('Label for mortality')
        
        df_pos = df[df['DEST'].contains('died')] 
        df_neg = df[~df['DEST'].contains('died')]

    elif label_type == 'ICU':
        print('label for ICU admission')

        df_pos = df[df['DEPARTMENT'].contains('ICU')] 
        df_neg = df[~df['DEPARTMENT'].contains('ICU')]

    print('pos df:',df_pos.shape, '-->',len(df_pos['PATIENTNR'].unique()), 'patients')
    print('neg df:',df_neg.shape, '-->',len(df_neg['PATIENTNR'].unique()), 'patients')

    print('-----Sampling for positive patients-----') 

    count = 0 

    for idx in np.unique(df_pos['ID']): # loop over patients
        # print(idx)
        patient = df_pos[df_pos['ID']==idx].sort_values(by='TIME').reset_index(drop=True) # Extract data of single patient, sort by date

        if label_type == 'ICU':
            t_event = patient[patient['DEPARTMENT']=='IC']['TIME'].min() # define moment of ICU admission as first ICU measurement
        elif label_type == 'mortality':
            t_event = patient['TIME'].max() 
            
            
        if (t_event - patient['TIME'].min()).total_seconds()/3600 < gap: # cannot label patients for which time between start and event is shorter than the gap
            count += 1
        else:

            #For positive feature vectors of positive patient
            ts = []            
            t = t_event - timedelta(hours=gap) #initialize t
            window = pred_window

            for i in range(int(window/int_pos)-1): # Make array with timestamps to sample from, making steps of size 'int_pos'
                ts.append(t)
                t = t - timedelta(hours=int_pos)

            for t in ts: 

                temp = patient[patient['TIME'] <= t].reset_index(drop=True)

                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,df_train,df_demo,df_demo_train,feature_window,features,idx)
                pos.append(v)

            # For Negative feature vectors of positive patients
            ts = []            
            t = t_event - timedelta(hours=gap+pred_window) #initialize t 
            window = (t_event - patient['TIME'].min()).total_seconds()/3600 - pred_window - gap # window to sample from is full window - prediction window and gap

            for i in range(int(window/int_neg)-1): # Make array with timestamps to sample from, making steps of size 'int_neg'
                ts.append(t)
                t = t - timedelta(hours=int_neg)

            for t in ts: 

                temp = patient[patient['TIME'].dt.date <= t].reset_index(drop=True) # Extract snippet before this day

                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,df_train,df_demo,df_demo_train,feature_window,features,idx)
                neg.append(v)

    print('-----Sampling for negative patient-----')

    for idx in np.unique(df_neg['ID']): # loop over patients
        # print(idx)
        patient = df_neg[df_neg['ID']==idx].sort_values(by='TIME').reset_index(drop=True) # Extract data of single patient, sort by date

        if (patient['TIME'].max() - patient['TIME'].min()).total_seconds()/3600 < gap: # cannot label patients with stay shorter than the gap
            count+= 1

        else:
            ts = []            
            t = patient['TIME'].max() #initialize t 
            window = (patient['TIME'].max() - patient['TIME'].min()).total_seconds()/3600 # window to sample from is full window 

            for i in range(int(window/int_neg)-1): # Make array with timestamps to sample from, making steps of size 'int_neg'
                ts.append(t)
                t = t - timedelta(hours=int_neg)

            for t in ts: 

                temp = patient[patient['TIME'].dt.date <= t].reset_index(drop=True) # Extract snippet before this day

                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,df_train,df_demo,df_demo_train,feature_window,features,idx)
                neg.append(v)

    print('number of patients with too little data for feature vector: ', count)            

    pos=np.array([np.array(x) for x in pos])
    neg=np.array([np.array(x) for x in neg])
    print('shape of positive class: ', pos.shape, 'shape of negative class: ', neg.shape)

    X = np.concatenate((pos, neg), axis=0)
    y = np.concatenate((np.ones(pos.shape[0]),np.zeros(neg.shape[0])),axis=0)

    print(X.shape)
    assert(np.isnan(X).any() == False)
    print(y.shape)
    assert(np.isnan(y).any() == False)

    return X, y     
   
