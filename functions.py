# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:23:05 2020

@author: 777018
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import sys
# import numpy
np.set_printoptions(threshold=sys.maxsize)


def importer_cci(file):
    df = pd.read_excel(file, header=0)
    
    type_lib = {
                'ID':int,
                'SEKS':float,
                'AGE':float,
                'LIM': float,
                'NIV':float,
                }
    df = df.astype(type_lib)
    
    return df

def importer_pacmed(features,specs):
    from data_warehouse_utils.dataloader import DataLoader
    dl = DataLoader()

    #find common parameters
    patients = dl.get_episodes()
    
    
    
    B = dl.get_single_timestamp(
                                parameters=features,
    #                             hospitals = ['franciscus']
                                columns = list(['hash_patient_id','pacmed_name','effective_timestamp','numerical_value','is_correct_unit_yn','unit_name'])
                               )

    # filter those with double episodes    
    idxs = []
    for idx in np.unique(patients.hash_patient_id):
        if len(np.unique(patients[patients['hash_patient_id']==idx]['episode_id']))>1:
            idxs.append(idx)
    idxs = np.asarray(idxs)
    print('N patients withg double episode to be filtered:', len(idxs))
    mask = patients['hash_patient_id'].isin(idxs)
    patients = patients[~mask]
    
    
    A = patients
    print(A.shape)
    # first filter transfer and still_admitted patients
    mask = A['outcome']=='transfer' 
    A = A[~mask]
    print(A.shape)
    mask = A['outcome']=='still_admitted' 
    A = A[~mask]
    print(A.shape)
    
    # for positive group, keep only those in 'icu_mortality', not in 'mortality'
    pos = A.copy()
    mask = pos['icu_mortality']
    pos = pos[mask]
    
    #filter those for which last discharge timestamp and death timestamp do not match
    mask = abs((pos['death_timestamp'] - pos['discharge_timestamp']).dt.days) >1
    pos = pos[~mask]
    
    
    #keep only informative columns:
    pos = pos[['hash_patient_id','age','bmi','gender','discharge_timestamp','icu_mortality']]
    print(pos.shape)
    
    # for negative group, only keep 'survivors'
    neg = A.copy()
    mask = neg['outcome'] == 'survivor'
    neg = neg[mask]
    
    neg = neg[['hash_patient_id','age','bmi','gender','discharge_timestamp','icu_mortality']]
    print(neg.shape)
    
    full = pd.concat([pos,neg],axis=0)
    full.columns = ['ID','AGE','BMI','SEX','t_event','mortality']
    print(full.shape)
    
    ## Filter B matrix
    
    print(B.shape)
    
    # Keep only those which ID is also present in A table (otherwhise no label available)
    mask = B['hash_patient_id'].isin(full['ID'].values)
    B = B[mask]
    print('after removing data points without label available',B.shape)
    
    #filter those without non-numerical value
    mask  = B.numerical_value.isnull()
    B = B[~mask]
    print('After removing those without numerical value',B.shape)
    
    
    # remove those without validated unit
    B.loc[B['is_correct_unit_yn'].isna(), ['is_correct_unit_yn']] = True
    mask = B['is_correct_unit_yn']
    B = B[mask]
    print('After removing unvalidated units:',B.shape)
    
    # remove rows for which no timestamp is available
    mask = B['effective_timestamp'].isnull()
    B = B[~mask]
    print('After removing unavailable dates:',B.shape)
    
    # ---- Make Unit dictionary ---------
    units_vitals = list(['[%]','[bpm]','[mmHg]','[/min]','[°C]'])
    units = []
    all_units = list(features)
    
    for i in all_units:
        snip = B[B['pacmed_name']==i].reset_index(drop=True)['unit_name'].dropna()
        if snip.shape[0] > 0:
            units.append('[' + snip[0] + ']')
        else:
            units.append(' ')
    
    dict_units = dict(zip(all_units, units))
    
    all_units = np.asarray(all_units)
    
    if specs['freq']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_freq'))
        dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_freq'))
        dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
            
    if specs['inter']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_inter'))
        dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_inter'))
        dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
    if specs['diff']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_diff'))
        dict_units.update(dict(zip(new_features, units)))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_diff'))
        dict_units.update(dict(zip(new_features, units_vitals)))    
        
    if specs['stats']:
        
        stats = ['_max','_min','_mean','_median','_std','_diff_std']
        for stat in stats:
            new_features = []
            for i in ['SpO2','HR','BP','RR']:
                new_features.append(i+stat)
                dict_units.update(dict(zip(new_features, units_vitals[:-1])))    
    
    # default_data.update({'item3': 3})
    
    dict_units.update({'BMI':''})
    dict_units.update({'AGE':'[yrs]'})
    dict_units.update({'SEX':'[yrs]'})
    dict_units.update({'SpO2':'[%]'})
    dict_units.update({'HR':'[bpm]'})
    dict_units.update({'BP':'[mmHg]'})
    dict_units.update({'RR':'[/min]'})
    dict_units.update({'Temp':'[°C]'})
    dict_units.update({'LOS':'[hrs]'})
    
    
    # get rid of non-informative columns, transform datatypes
    B = B[['hash_patient_id','pacmed_name','effective_timestamp','numerical_value']]
    type_lib = {
                    'hash_patient_id': 'category',
                    'pacmed_name':'category',
                    'effective_timestamp':np.datetime64,
    #                 'numerical_value':np.float16,
                    }
    B = B.astype(type_lib)
    
    #import vitals and merge with B:
    vitals = pd.read_csv("../vitals_update_sliding.csv", index_col=0)
    
    vitals.columns = ['ID','VAR','TIME','VAL']
    
    #filter those without non-numerical value
    mask  = vitals.VAL.isnull()
    vitals = vitals[~mask]
    print('After removing those without numerical value',vitals.shape)
        
    # remove rows for which no timestamp is available
    mask = vitals['TIME'].isnull()
    vitals = vitals[~mask]
    print('After removing unavailable dates:',vitals.shape)
    
    B.columns = ['ID','VARIABLE','TIME','VALUE']
    vitals.columns = ['ID','VARIABLE','TIME','VALUE']
    B_full = pd.concat([B,vitals],axis=0)
    
    
    B_full.columns = ['ID','VARIABLE','TIME','VALUE']
    print('IDS in single timestamp:', len(np.unique(B_full.ID)))
    print('IDS in episodes:', len(np.unique(full.ID)))
    
    ids = np.unique(full.ID)
    mask = B_full['ID'].isin(ids)
    B_full = B_full[mask]
    
    print('IDS in single timestamp after removing still admitted and transfer:', len(np.unique(B_full.ID)))
    
    ids = np.unique(B_full.ID)
    mask = full['ID'].isin(ids)
    full = full[mask]
    
    
    
    assert len(np.unique(B_full.ID)) == len(np.unique(full.ID))
    print(B_full.info())
    
    
    
    
    return B_full, full,dict_units
    
def importer_labs(file,encoding,sep,header,specs,labs=True,filter=True,nrows=None,skiprows=None,skipinitialspace=False):
    """Import labs data.

        Parameters
        ----------
        file : [str]
            `hash_patient_id`s of patients to query.
        columns : Optional[List[str]]
            List of columns to return.

        Returns
        -------
        type : pd.DataFrame
        """


    print('importer labs triggered')
    if labs:
        col_list = ['PATIENTNR','OMSCHRIJVING','BMI','LEEFTIJD',
                    'OPNAMEDATUM','ONTSLAGDATUM','HERKOMST',
                    'BESTEMMING',
                # 'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
                'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','UNIT'
                ]

    else:

            col_list = ['PATIENTNR', 'MEETMOMENT',  'METINGTYPELABEL','VOLGNUMMER',
            'Data1', 'Data2', 'Data3', 'INVOERMOMENT',
            'METINGCATEGORIE']
    if filter:
        usecols = [0,1,2,3,
                    4,5,7,
                   9,
                   # 10,11,
                   13,14,17,18,19]
    else:
        usecols = None
    
    type_lib = {
                'PATIENTNR':str,
                'OMSCHRIJVING':'category',
                'BMI':str,
                # 'LEEFTIJD': int,
                'OPNAMEDATUM':str,
                'ONTSLAGDATUM': str,
                'HERKOMST':'category',
                'BESTEMMING':'category',
                # 'DOSSIER_BEGINDATUM':str,
                # 'DOSSIER_EINDDATUM':str,
                'OPNAMETYPE':'category',
                'AFNAMEDATUM':str,
                'DESC':'category',
                'UNIT':str,
                # 'UITSLAGDATUM':str,
                # 'RESERVE':str,
                
                }
    
    data =  pd.read_csv(file,
                        # engine='python',
                        sep=sep,
                        # lineterminator='\r',
                        # error_bad_lines=False,
                        # warn_bad_lines=True,
                        encoding=encoding,
                          index_col=False,
                         # sheet_name = 'Sheet1',
                            header=header,
                            usecols = usecols,
                            dtype=type_lib,
                            # nrows = nrows,
                            # skiprows=skiprows,
                            # skipinitialspace=skipinitialspace
                            # delim_whitespace=True
      )

    print('Lab Data imported')
    # print()

    # Make DF with unique labs and corresponding units
    

    
    
    if filter:
        data.columns = col_list
    else: 
        data = data[col_list]
    
    data = data.astype(type_lib)
     
    # ---- Make Unit dictionary ---------
    units_vitals = list(['[%]','[bpm]','[mmHg]','[/min]','[°C]'])
    units = []
    all_units = list(data['DESC'].unique())
    
    for i in all_units:
        snip = data[data['DESC']==i].reset_index(drop=True)['UNIT'].dropna()
        if snip.shape[0] > 0:
            units.append('[' + snip[0] + ']')
        else:
            units.append(' ')
    
    dict_units = dict(zip(all_units, units))
    
    all_units = np.asarray(all_units)
    
    if specs['freq']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_freq'))
        dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_freq'))
        dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
            
    if specs['inter']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_inter'))
        dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_inter'))
        dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
    if specs['diff']:
        new_features = []
        for i in all_units:
            new_features.append(i+str('_diff'))
        dict_units.update(dict(zip(new_features, units)))    
        
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:
            new_features.append(i+str('_diff'))
        dict_units.update(dict(zip(new_features, units_vitals)))    
        
    if specs['stats']:
        
        stats = ['_max','_min','_mean','_median','_std','_diff_std']
        for stat in stats:
            new_features = []
            for i in ['SpO2','HR','BP','RR']:
                new_features.append(i+stat)
                dict_units.update(dict(zip(new_features, units_vitals[:-1])))    
    
    # default_data.update({'item3': 3})
    
    dict_units.update({'BMI':''})
    dict_units.update({'AGE':'[yrs]'})
    dict_units.update({'SpO2':'[%]'})
    dict_units.update({'HR':'[bpm]'})
    dict_units.update({'BP':'[mmHg]'})
    dict_units.update({'RR':'[/min]'})
    dict_units.update({'Temp':'[°C]'})
    dict_units.update({'LOS':'[hrs]'})
    
    
    print('UNIQUE IDs lab data:', len(np.unique(data['PATIENTNR'])))

    return data, dict_units


def importer_vitals(file,encoding,sep,header,):
    print('importer vitals triggered')
    extract = [0,1,5,7,8,9,10,11,13]
    # PATIENTNR,SUBJECTNR
    col_list = ['SUBJECTNR','OMSCHRIJVING', 'MEETMOMENT', 'METINGTYPELABEL',
    'Data1', 'Data2', 'Data3','Eenheid', 'METINGCATEGORIE']
    
    new_cols = ['PATIENTNR','OMSCHRIJVING', 'AFNAMEDATUM', 'DESC',
    'UITSLAG', 'Data2', 'Data3', 'UNIT','OPNAMETYPE']
    
    type_lib = {'SUBJECTNR':str,
                'OMSCHRIJVING':'category',
                'MEETMOMENT': str,
                'METINGTYPELABEL': 'category',
                'Data1':str,
                'Data2':str,
                'Data3':str,
                'Eenheid':str,
                'METINGCATEGORIE': 'category'
                
                }
    
    data =  pd.read_csv(file,
                        # engine='python',
                        sep=sep,
                        # lineterminator='\r',
                        # error_bad_lines=False,
                        # warn_bad_lines=True,
                        encoding=encoding,
                          index_col=False,
                         # sheet_name = 'Sheet1',
                            header=header,
                            # usecols = usecols,
                            dtype=type_lib,
                            # nrows = nrows,
                            # skiprows=skiprows,
                            # skipinitialspace=skipinitialspace
                            # delim_whitespace=True
      )
    data=data[col_list]
    print('Vitals Data imported')

    data.columns = col_list
    data = data.astype(type_lib)
    data.columns = new_cols
    print('UNIQUE IDs vitals data:', len(np.unique(data['PATIENTNR'])))
    return data
    
def cleaner_labs(data):
    print('cleaner labs triggered')
    data.info()
    

    #BMI
    data['BMI'] = data['BMI'].apply(lambda x: x.replace(',','.'))
    data['BMI'] = data['BMI'].astype(float)
    #Dates
    date_format='%Y-%m-%d %H:%M:%S'
    

    dates = [
        # 'OPNAMEDATUM','ONTSLAGDATUM','DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM','UITSLAGDATUM',
             'AFNAMEDATUM']
    for i in dates:
        print(i)
        mask = data[i].str.startswith('29')
        data = data[~mask]
        data[i]= data[i].str[:19]
        # print(data[i][1])
        # print(type(data[i][1]))
        # print(len(data[i][1]))
        data[i] = pd.to_datetime(data[i],format = date_format)
        print('done')
        data[i] = pd.to_datetime(data[i],format=date_format)
    
    type_lib = {
                'PATIENTNR':str,
                'OMSCHRIJVING':'category',
                'BMI':np.float16,
                'LEEFTIJD': str,
                # 'HERKOMST':'category',
                'BESTEMMING':'category',
                'OPNAMETYPE':'category',
                'DESC':'category',
                # 'EENHEID':'category',
                'UITSLAG':str
                }
    data = data.astype(type_lib)
    
    #clean PATIENTNR

    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('sub', '', regex=True)
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('@', '', regex=True)
    
    mask = data['PATIENTNR'].str.contains('nan')
    data = data[~mask]
    
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].astype(float)
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].astype(int)
    
    
    # clean 'non informative' features --> non-numerics
    data['UITSLAG'] = pd.to_numeric(data['UITSLAG'], errors='coerce')
    print(data.shape)
    data = data[data['UITSLAG'].notna()]
    print(data.shape)
                                                                                                                                                                                     
    
    # labels
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('PUK','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Dialyse','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Dagverpleging','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Anders klinisch','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Observatie','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Gastverblijf','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Afwezigheid','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Verkeerd bed','Klinische opname')
    
    
    # trasform UITSLAG in floats
    data['UITSLAG'].astype(float)
    
    # identify PIDs without known OPNAMETYPE or BESTEMMING
    ids_unknown = []
    for i in data['PATIENTNR'].unique():
        ex = data[data['PATIENTNR']==i].reset_index()
        if ex['OPNAMETYPE'].isnull().all(): # all type unknown
            # print(ex)
            ids_unknown.append(ex['PATIENTNR'][0])
        elif ex['BESTEMMING'].isnull().all():   # all bestemming unknown
            ids_unknown.append(ex['PATIENTNR'][0])
        elif ex['BESTEMMING'].iloc[-1] == 'Onbekend': # last bestemming unknown
            ids_unknown.append(ex['PATIENTNR'][0])
            
    all_ids = np.unique(data['PATIENTNR'])
    ids = [x for x in all_ids if x not in ids_unknown]
    data = data[data['PATIENTNR'].isin(ids)] # filter out PIDs without known OPNAMETYPE or BESTEMMING
    
    data.info()
    pids = data.PATIENTNR.unique()
    
    
    covid_statuses = list(data['OMSCHRIJVING'].unique())
    
    for idx in pids:
        ex = data[data['PATIENTNR']==idx]
        statuses = list(ex['OMSCHRIJVING'])
        if 'COVID-19 / SARS-CoV-2' in statuses and 'Klinisch COVID-19 / SARS-CoV-2 PCR negatief' in statuses:
            print('patient ID with both pos and neg covd:',idx)    
    
    # keep Only covid positive (!) in df
    mask = (data['OMSCHRIJVING']=='COVID-19 / SARS-CoV-2') + (data['OMSCHRIJVING']=='Klinisch COVID-19 / SARS-CoV-2 PCR negatief')
    df_lab = data[mask]
    
    # check unknown OPNAMETYPE
    mask = df_lab['OPNAMETYPE'].isna()
    df_lab = df_lab[~mask]
    df_lab = df_lab.reset_index(drop=True)
    
    

    
    print('Lab Data cleaned')
    
    return df_lab

def cleaner_vitals(data):
    print('cleaner vitals triggered')
    
    #filter non-numeric values
    mask = data['UITSLAG'].str.isalpha()
    data = data[~mask]
    mask = data['UITSLAG'].isna()
    data = data[~mask]
    print(data.shape)
    data = data[data.UITSLAG.apply(lambda x: x.isnumeric())]
    print(data.shape)
       
    
    # filter columns with mean and diastolic BP
    data = data.drop(['Data2' , 'Data3'] , axis='columns')
    
    #clean PATIENTNR

    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('sub', '', regex=True)
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('@', '', regex=True)
    
    mask = data['PATIENTNR'].str.contains('nan')
    data = data[~mask]
    
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].astype(float)
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].astype(int)
    
    for i in range(data.shape[0]):
        
        try:
            data['UITSLAG'].iloc[i] = float(data['UITSLAG'].iloc[i])
        except:
            print(data['UITSLAG'].iloc[i])
            
    # data['Data2'] = data['Data2'].astype(np.float16)
    # data['Data3'] = data['Data3'].astype(np.float16)
    data['AFNAMEDATUM'] = data['AFNAMEDATUM'].str[:19]
    data['AFNAMEDATUM'] = pd.to_datetime(data['AFNAMEDATUM'],format='%Y-%m-%d %H:%M:%S')
    
    data['DESC'] = data['DESC'].str.replace('ABP','BP')
    data['DESC'] = data['DESC'].str.replace('NIBP','BP')
    data['DESC'] = data['DESC'].str.replace('Resp(vent)','RespVent')
    data['DESC'] = data['DESC'].str.replace('Resp','RR')
   
    print(data['OPNAMETYPE'].unique())
    print('UNIQUE OPNAMETYPE VITALS: \n', data['OPNAMETYPE'].value_counts())
    
    data = data[data['OPNAMETYPE']=='USER']
    # data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('ONBEKEND','USER')
    
    print(data.shape)
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('USER','Klinische opname')
    
    print('Unique types in vitals set:',data['OPNAMETYPE'].unique())
    df_vitals = data.reset_index(drop=True)
    print('Vitals Data cleaned')
    return df_vitals


def df_merger(df_1,df_2,df_cci,specs):
    print('df_merger triggered')
    
    print(df_1['OPNAMETYPE'].value_counts())
    print(df_2['OPNAMETYPE'].value_counts())
    
    df = pd.concat([df_1,df_2],axis=0)
    
    print('----Feature selection-----')
    counts = df_1['DESC'].value_counts()
    features = counts.index[:specs['n_features']]
    
    print(len(features),' lab features included')
    
    # print('Vitals features ranking:')
    print(df_2['DESC'].value_counts())
    
    # features_vitals = df_2['DESC'].value_counts().index
    # freqs = df_2['DESC'].value_counts().values
    # idx = np.where(freqs > 0.8*freqs[0])
    # features_vitals = features_vitals[idx]
    
    features_vitals = ['SpO2','HR','BP','RR','Temp']
    print(len(features_vitals),' vitals features included')
    
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    # features = np.asarray(features_vitals)

    print('N features (Unique vitals + labs): ', features.shape)
    mask = df['DESC'].isin(features)
    df = df[mask]

    
    col_list = ['PATIENTNR','BMI','LEEFTIJD','BESTEMMING',
            'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG']
    df = df[col_list]

    df.columns = ['ID','BMI','AGE','DEST',
            'DEPARTMENT','TIME','VARIABLE','VALUE']
    
    
    ids = np.unique(df['ID']) # Unique IDs in dataset
    print('N patients:',len(ids))
    # filter 'Outpatient data'
    outpats = []
    for i in ids:
        snip = df[df['ID']==i]
        snip = snip.sort_values(by='TIME')
        dates = snip['TIME'].unique()

        if dates.shape[0]>1:
            d = (dates[-1] - dates[-2])
            days = d.astype('timedelta64[D]')
            outpats.append((days / np.timedelta64(1, 'D')))
        else:
            outpats.append(0)
    outpats = np.asarray(outpats)
    ids_out = ids[np.where(outpats>3)]
    print('Identified óutpatient patients: \n',ids_out)
    ids_remove = []
    
    rest = pd.DataFrame()
    for i in ids_out:
        snip = df[df['ID']==i].sort_values(by='TIME')
        df = df[df['ID']!=i]
        # print(df.shape)
        dates = snip['TIME'].unique()
        dates = dates[:-1]
        
        mask = snip['TIME'].isin(dates)
        snip = snip[mask]
        if snip.shape[0]<3:
            ids_remove.append(i)
        rest = pd.concat([rest,snip],axis=0)
    
    df = pd.concat([df,rest],axis=0)
    print('PIDs removed for too little data:', ids_remove)
    
    mask = df['ID'].isin(ids_remove)
    df = df[~mask]
    print('n patients left after removed for too little data:',len(df['ID'].unique()))
    
    if specs['policy']:
        print('Filter NO-IC policy patients')
        # Filter No-ICU policy patients
        
        pol_idx = df_cci[(df_cci['LIM'] == 1)|(df_cci['LIM'] == 5)]['ID'].values
        
        print('n patients left:',len(df['ID'].unique()))
        mask =  pd.Series(ids).isin(pol_idx)
        ids = ids[~mask] #remove IDs which have no IC policy
        print('n patients left after filtering no ICU policy:', len(df['ID'].unique()))
        mask = df['ID'].isin(ids)
        df = df[mask]
    
    print(df.shape)
    

    
    print('Unique features:', df['VARIABLE'].unique())
    print('Unique patients:', len(df['ID'].unique()))
    return df,features




def get_ids(df):
    print('get_ids triggered')
    # get IDs of all ICU containing patients
    ids_IC = []
    for i in df['ID'].unique():
        opnametypes = df[df['ID']==i].sort_values(by='TIME')
        opnametypes = opnametypes.DEPARTMENT  # filter out types of individual patient
        
        if len(opnametypes) == 0:
            print('Patient ', i, ' contains no type info')
        if opnametypes.str.contains('IC').sum() > 0:    # check if it contains any ICU admission
            ids_IC.append(i)
    
    # get idx of IC which also seen the clinic
    ids_events = []
    for i in ids_IC:
        opnametypes = df[df['ID']==i].sort_values(by='TIME')
        opnametypes = opnametypes['DEPARTMENT'].dropna().reset_index(drop=True) # filter out types of individual patient
        # print(opnametypes)
        if len(opnametypes) == 0:
            print('Patient ', i, ' contains no type info')
        if opnametypes[0] == 'Klinische opname':
            ids_events.append(i)
    
    
    ids_IC_only = [x for x in ids_IC if x not in ids_events]
    ids_all = df['ID'].unique()
    ids_clinic = [x for x in ids_all if x not in ids_IC_only]
    ids_clinic = [x for x in ids_clinic if x not in ids_events]
    
    assert len(ids_all) == len(ids_IC_only) + len(ids_events) + len(ids_clinic)
    
    print(len(ids_all),'unique patients','\n Only clinic: ',len(ids_clinic),
          '\n Only ICU: ',len(ids_IC_only),'\n Both: ',len(ids_events))
        
    return ids_IC_only, ids_all, ids_clinic, ids_events


def missing(df,features,ids_clinic,ids_events,x_days = True):
    
    from datetime import datetime,timedelta
    x = 5
    
    df = df[['ID','VARIABLE','TIME','VALUE','DEPARTMENT']]
    df = np.asarray(df)
    df1 = []
    n_days_clinic = []
    # clinic only
    for idx in ids_clinic:

        patient = df[np.where(df[:,0]==idx)]
        patient = patient[patient[:,2].argsort()]
            
        if x_days:    
            t_start = patient[0,2]# define moment of ICU admission as first ICU measurement
            t = t_start + timedelta(hours=x*24) 
            patient = patient[np.where(patient[:,2]<t)]
            
        n_days = np.round((patient[-1,2] - patient[0,2]).total_seconds() / 3600.0/24)
        if n_days == 0:
            n_days = 1
        n_days_clinic.append(n_days)    
        for feature in features:
            temp = patient[np.where(patient[:,1]==feature)]
            v = []
            v.append(idx)
            v.append(feature)
            v.append(temp.shape[0]/n_days)
            df1.append(v)
    
    df1=np.array([np.array(x) for x in df1])
    df1 = pd.DataFrame(df1)
    df1.columns = ['ID','feature','/day']
    df1['label'] = 'clinic'
    
    
    df2 = []
    
    n_days_events = []
    # clinic + ICU
    for idx in ids_events:
        
        patient = df[np.where(df[:,0]==idx)]
        patient = patient[patient[:,2].argsort()]
        t_event = patient[np.where(patient[:,4]=='IC')][0,2]# define moment of ICU admission as first ICU measurement
        patient = patient[np.where(patient[:,2]<t_event)]
        
        if x_days:    
            t_start = patient[0,2]# define moment of ICU admission as first ICU measurement
            t = t_start + timedelta(hours=x*24) 
            patient = patient[np.where(patient[:,2]<t)]
        
        
        n_days = np.round((patient[-1,2] - patient[0,2]).total_seconds() / 3600.0/24)
        
        if n_days == 0:
            n_days = 1
        n_days_events.append(n_days)    
        for feature in features:
            temp = patient[np.where(patient[:,1]==feature)]
            v = []
            v.append(idx)
            v.append(feature)
            v.append(temp.shape[0]/n_days)
            df2.append(v)
    
    df2=np.array([np.array(x) for x in df2])
    df2 = pd.DataFrame(df2)
    df2.columns = ['ID','feature','/day']
    df2['label'] = 'clinic+IC'
    
    df_full = pd.concat([df1,df2])
    
    n_clinic =np.array([np.array(x) for x in n_days_clinic])
    n_events = np.array([np.array(x) for x in n_days_events])
    return df_full,n_clinic,n_events
            
def df_preparer_pacmed (df,df_episodes,random_state,specs,norm=True):
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    df['TIME'] = pd.to_datetime(df['TIME'],format='%Y-%m-%d %H:%M:%S')
    
    
    ids = np.unique(df['ID']) # Unique IDs in dataset
    mask = df_episodes['ID'].isin(ids)
    df_episodes = df_episodes[mask]
    
    df_demo = df_episodes[['ID','AGE','BMI','SEX']]
    df_demo['SEX'] = df_demo['SEX'].replace('V', 0)
    df_demo['SEX'] = df_demo['SEX'].replace('M',1)
    
    print(' TOTAL ids:', len(ids))
    
    ids_events = df_episodes[df_episodes['mortality']==True]['ID'].values
    y = np.in1d(ids,ids_events)
    variables = np.unique(df['VARIABLE'])
    
    ids_train,ids_val = train_test_split(ids, test_size=specs['val_share'],random_state=random_state,stratify=y)
   
    
    df_train_full = df[df['ID'].isin(ids_train)]
    df_val = df[df['ID'].isin(ids_val)]
    
    
    df_demo_train_full = df_demo[df_demo['ID'].isin(ids_train)]
    df_demo_val = df_demo[df_demo['ID'].isin(ids_val)]

    #split training set in training and testing set
    ids = np.unique(df_train_full['ID'])
    y = np.in1d(ids,ids_events)
    ids_train,ids_test = train_test_split(ids, test_size=specs['test_share'],random_state=random_state,stratify=y)
    
    print( len(ids))
    print(len(ids_train)+len(ids_test)+len(ids_val))
    
    
    df_train = df_train_full[df_train_full['ID'].isin(ids_train)]
    df_test = df_train_full[df_train_full['ID'].isin(ids_test)]

    df_demo_train = df_demo_train_full[df_demo_train_full['ID'].isin(ids_train)]
    df_demo_test = df_demo_train_full[df_demo_train_full['ID'].isin(ids_test)]
    
    
    
    print('Split train, val en test set: \n original shape: ',df.shape,
          '\n train shape: ',df_train.shape, 'unique patients: ', len(df_train['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_train['ID']),ids_events)),
          '\n Val shape: ',df_val.shape, 'unique patient: ', len(df_val['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_val['ID']),ids_events)),
          '\n Test shape: ',df_test.shape, 'unique patients: ', len(df_test['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_test['ID']),ids_events))
          )
    
    
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
        
        df_demo_train_norm['SEX'] =  df_demo_train['SEX']
        df_demo_val_norm['SEX'] =  df_demo_val['SEX']
        df_demo_test_norm['SEX'] =  df_demo_test['SEX']
        
        df_demo_train = df_demo_train_norm
        df_demo_val = df_demo_val_norm
        df_demo_test = df_demo_test_norm
            
        print('data normalized using standardscaler')
        
        

    return df_train,df_val,df_test,df_demo_train,df_demo_val,df_demo_test
    

def df_preparer(df,variables,ids_ICU_only,ids_events,random_state,specs,norm=True):
    
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
    ids_ICU_only: np.array[int]
        Array with pateint IDs which only have been in ICU admission (so need to be skipped)
    ids_events: np.array[int]
        Array with pateint IDs which are positive class
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
   
    
    #Filter ICU only patients
    mask = df['ID'].isin(ids_ICU_only)
    df = df[~mask]
    print('N patients in df after filtering only ICU:',len(np.unique(df['ID'])))
    
    
    
    ids = np.unique(df['ID']) # Unique IDs in dataset
        
    # create df with Demographics data (BMI and AGE) -- >  df_demo
    bmi = []
    age =[]
    for i in ids:
        temp = df[df['ID']==i]
        mask =  temp['BMI'].isna()
        if temp[~mask].shape[0]>0:
            bmi.append(temp[~mask].reset_index()['BMI'][0])
        else:
            bmi.append(np.nan)
        temp = df[df['ID']==i]
        mask =  temp['AGE'].isna()
        if temp[~mask].shape[0]>0:
            age.append(float(temp[~mask].reset_index()['AGE'][0]))
        else:
            age.append(np.nan)
    df_demo = pd.DataFrame()
    df_demo['ID'] = ids
    df_demo['BMI'] = bmi
    df_demo['AGE'] = age
    
    # Outlier detection BMI and Age
    bmi = df_demo['BMI']
    m_bmi = np.mean(bmi[~bmi.isnull()])
    s_bmi = np.std(bmi[~bmi.isnull()])
    print(m_bmi-3*s_bmi,m_bmi+3*s_bmi)
    
    df_demo.loc[(df_demo['BMI'] < (m_bmi - 3*s_bmi)) | (df_demo['BMI'] > (m_bmi + 3*s_bmi)), 'BMI'] = np.nan
    
    age = df_demo['AGE']
    m_age = np.mean(age[~age.isnull()])
    s_age = np.std(age[~age.isnull()])
    df_demo.loc[(df_demo['AGE'] < m_age - 3*s_age) | (df_demo['AGE'] > m_age + 3*s_age), 'AGE'] = np.nan
    print(m_age-3*s_age,m_age+3*s_age)
    
    # Make figure of demographics transfer group vs non-transfer
    # df_demo_plot = df_demo.copy()
    # df_demo_plot['label'] = 'no transfer'
    # mask = df_demo_plot['ID'].isin(ids_events)
    # df_demo_plot.loc[mask,'label'] = 'transfer'
    
    # df_1 = df_demo_plot[['BMI','label']]
    # df_1.columns = ['Value','label']
    # df_1['feature'] = 'BMI'
    
    # df_2 = df_demo_plot[['AGE','label']]
    # df_2.columns = ['Value','label']
    # df_2['feature'] = 'AGE [yrs]'
    
    # df_full = pd.concat([df_1,df_2],axis=0)
    
    # import seaborn as sns
    
    
    # # Draw a nested boxplot to show bills by day and time
    # ax = sns.boxplot(x="feature", y="Value",
    #             hue="label", palette=["b", "r"],
    #             data=df_full)
    
    # plt.legend(bbox_to_anchor=(1, 1), loc=2) 
    # plt.tight_layout()
    # plt.savefig('Demo.png',dpi=300)
    
    print(' TOTAL ids:', len(ids))
    # Split raw df in training and validation set on patient level:
    y = np.in1d(ids,ids_events) #make label vector
    
    ids_train,ids_val = train_test_split(ids, test_size=specs['val_share'],random_state=random_state,stratify=y)
   
    
    df_train_full = df[df['ID'].isin(ids_train)]
    df_val = df[df['ID'].isin(ids_val)]
    
    
    df_demo_train_full = df_demo[df_demo['ID'].isin(ids_train)]
    df_demo_val = df_demo[df_demo['ID'].isin(ids_val)]

    #split training set in training and testing set
    ids = np.unique(df_train_full['ID'])
    y = np.in1d(ids,ids_events)
    ids_train,ids_test = train_test_split(ids, test_size=specs['test_share'],random_state=random_state,stratify=y)
    
    # print( len(ids))
    # print(len(ids_train)+len(ids_test)+len(ids_val))
    
    
    df_train = df_train_full[df_train_full['ID'].isin(ids_train)]
    df_test = df_train_full[df_train_full['ID'].isin(ids_test)]

    df_demo_train = df_demo_train_full[df_demo_train_full['ID'].isin(ids_train)]
    df_demo_test = df_demo_train_full[df_demo_train_full['ID'].isin(ids_test)]
    
    
    
    print('Split train, val en test set: \n original shape: ',df.shape,
          '\n train shape: ',df_train.shape, 'unique patients: ', len(df_train['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_train['ID']),ids_events)),
          '\n Val shape: ',df_val.shape, 'unique patient: ', len(df_val['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_val['ID']),ids_events)),
          '\n Test shape: ',df_test.shape, 'unique patients: ', len(df_test['ID'].unique()),'positives: ',sum(np.in1d(np.unique(df_test['ID']),ids_events))
          )
    
    
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




def Prepare_imputation_vectors(df_demo_train,df_train,features):
    #make median demo vector
    df_demo_train = df_demo_train[['BMI','AGE']]
    df_demo_train = df_demo_train.dropna()
    
    demo_median = np.asarray(df_demo_train.median())
    # print('median values for demograohics:',demo_median)
    
    #make median vector for other variables
    median = []
    for v in features:
        # print(v)
        m = np.median(df_train[df_train['VARIABLE']==v]['VALUE'].values)
        median.append(m)
    
    median = np.asarray(median)
    # print('median values:',median)
    
    return demo_median,median
    
def prepare_feature_vectors(df,median,df_demo,demo_median,df_episodes,ids_events,features,specs):
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
    y_pat = list()
    
    med_shares = []
    ff_shares = []
    
    med_shares_spec = []
    ff_shares_spec = []
    
      
        
    if specs['label_type'] == 'mortality':
        print('Label for mortality')
        mask = df_episodes.mortality
        ids_event = df_episodes[mask]['ID'].values        
        df_pos = df[df['ID'].isin(ids_event)] 
        df_neg = df[~df['ID'].isin(ids_event)]
    
        #Make arrays out of dfs, keep only ID, VARIABLE, TIME, VALUE, DEPARTMENT
        df_pos = df_pos[['ID','VARIABLE','TIME','VALUE']]
        df_neg = df_neg[['ID','VARIABLE','TIME','VALUE']]
        
        
    
    elif specs['label_type'] == 'ICU':
        print('label for ICU admission')

        df_neg = df[~df['ID'].isin(ids_events)] # Isolate negative df --> no ICU 
        df_pos = df[df['ID'].isin(ids_events)]
        

        # print('labels in negative df:',df_neg['DEPARTMENT'].unique())
        # print('labels in pos df:',df_pos['DEPARTMENT'].unique())
        
        #Make arrays out of dfs, keep only ID, VARIABLE, TIME, VALUE, DEPARTMENT
        df_pos = df_pos[['ID','VARIABLE','TIME','VALUE','DEPARTMENT']]
        df_neg = df_neg[['ID','VARIABLE','TIME','VALUE','DEPARTMENT']]
    
    print('pos df:',df_pos.shape, '-->',len(df_pos['ID'].unique()), 'patients')
    print('neg df:',df_neg.shape, '-->',len(df_neg['ID'].unique()), 'patients')
    
    df_pos = np.asarray(df_pos)
    df_neg = np.asarray(df_neg)
    df_demo = np.asarray(df_demo)

    # print('demographics shape:',df_demo.shape)
    
    print('-----Sampling for positive patients-----') 
    
    count = 0 
    
    for idx in np.unique(df_pos[:,0]): # loop over patients
        patient = df_pos[np.where(df_pos[:,0]==idx)]
        patient = patient[patient[:,2].argsort()]
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0]
        
        if specs['label_type'] == 'ICU':
            t_event = patient[np.where(patient[:,4]=='IC')][0,2]# define moment of ICU admission as first ICU measurement
         
        elif specs['label_type'] == 'mortality':
            t_event = patient[-1,2]
            
        if (t_event - patient[0,2]).total_seconds() / 3600.0 < specs['gap']: # cannot label patients for which time between start and event is shorter than the gap
            count += 1
        else:
            
            #For positive feature vectors of positive patient
            ts = []            
            t = t_event - timedelta(hours=specs['gap']) #initialize t
            window = specs['pred_window']
            los = np.round((t_event - patient[0,2]).total_seconds() / 3600.0,0) #initiate LOS variable [hours] (start with total stay, extract day accroding to interval)
            
            
            for i in range(int(window/specs['int_pos'])): # Make array with timestamps to sample from, making steps of size 'int_pos'
                ts.append(t)
                t = t - timedelta(hours=specs['int_pos'])

            
            if los < 0:
                    print(type(t_event))
                    print('wrong')
                    print('To ICU:',t_event)
                    print('first timepoint:',patient[0,2])
                    print(los)
                    
            # count_day = 1
            for t in ts:
                                    
                temp = patient[np.where(patient[:,2]<t)]
                
                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,median,demo,demo_median,features,los,specs)
                pos.append(v) # add feature vector to 'pos' list
                y_pat.append(1) # add patient label
                
                
                med_shares.append(med_share)
                ff_shares.append(ff_share)
                med_shares_spec.append(med_spec)
                ff_shares_spec.append(ff_spec)
                
                # if (count_day*int_pos)%24 == 0:
                los -= specs['int_pos']
                los = np.round(los,0)
                # count_day += 1
                
                
            # For Negative feature vectors of positive patients
            ts = []            
            t = t_event - timedelta(hours=specs['gap']+specs['pred_window']) #initialize t 
            los = np.round((t_event - patient[0,2]).total_seconds() / 3600.0,0) #initiate day variable (start with total stay, extract day according to interval)
            window = (t_event - patient[0,2]).total_seconds() / 3600.0 - specs['pred_window'] - specs['gap'] # window to sample from is full window - prediction window and gap
            
            for i in range(int(window/specs['int_neg'])): # Make array with timestamps to sample from, making steps of size 'int_neg'
                ts.append(t)
                t = t - timedelta(hours=specs['int_neg'])
                        
            if los < 0:
                    print('wrong')
                    print('To ICU:',t_event)
                    print('first timepoint:',patient[0,2])
                    print(los)
            
            # count_day = 1
            for t in ts:
                

                temp = patient[np.where(patient[:,2]<t)]
                
                
                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,median,demo,demo_median,features,los,specs)
                neg.append(v)# add feature vector to 'neg' list
                y_pat.append(1) # add patient label
                
                med_shares.append(med_share)
                ff_shares.append(ff_share)
                ff_shares.append(ff_share)
                med_shares_spec.append(med_spec)
                ff_shares_spec.append(ff_spec)
                
                # if (count_day*int_neg)%24 == 0:
                los -= specs['int_neg']
                los = np.round(los,0)
                # count_day += 1
                
    print('-----Sampling for negative patient-----')
    
    for idx in np.unique(df_neg[:,0]): # loop over patients
        patient = df_neg[np.where(df_neg[:,0]==idx)]
        patient = patient[patient[:,2].argsort()]
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0]
        t_event = patient[-1,2]
        if (patient[-1,2] - patient[0,2]).total_seconds() / 3600.0 < specs['gap']: # cannot label patients with stay shorter than the gap
            count+= 1

        else:
            ts = []            
            t = patient[-1,2] #initialize t 
            los = np.round((t_event - patient[0,2]).total_seconds() / 3600.0,0) #initiate day variable (start with total stay, extract day accroding to interval)
            window = (patient[-1,2] - patient[0,2]).total_seconds() / 3600.0 # window to sample from is full window 
            
            for i in range(int(window/specs['int_neg'])): # Make array with timestamps to sample from, making steps of size 'int_neg'
                ts.append(t)
                t = t - timedelta(hours=specs['int_neg'])
                
            if los < 0:
                    print('wrong')
                    print('To ICU:',t_event)
                    print('first timepoint:',patient[0,2])
                    print(los)
                       
            # count_day = 1
            for t in ts:
                
                temp = patient[np.where(patient[:,2]<t)]
                
                v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,median,demo,demo_median,features,los,specs)
                neg.append(v)# add feature vector to 'neg' list
                y_pat.append(0) # add patient label
                
                med_shares.append(med_share)
                ff_shares.append(ff_share)
                ff_shares.append(ff_share)
                med_shares_spec.append(med_spec)
                ff_shares_spec.append(ff_spec)
                
                # if (count_day*int_neg)%24 == 0:
                los -= specs['int_neg']
                los = np.round(los,0)
                # count_day += 1
                
                
    print('number of patients with too little data for feature vector: ', count)            
    print(len(pos),len(neg))
    pos=np.array([np.array(x) for x in pos])
    neg=np.array([np.array(x) for x in neg])
    y_pat = np.array([np.array(x) for x in y_pat])
    
    print('shape of positive class: ', pos.shape, 'shape of negative class: ', neg.shape)
    
    X = np.concatenate((pos, neg), axis=0)
    y = np.concatenate((np.ones(pos.shape[0]),np.zeros(neg.shape[0])),axis=0)
    assert(y.shape == y_pat.shape)
    
    imputation = [np.mean(med_shares), np.mean(ff_shares), (1-(np.mean(med_shares)+np.mean(ff_shares)))]
    feature_imputation = pd.DataFrame()
    feature_imputation['0'] = pd.DataFrame(med_shares_spec).mean(axis=0)
    feature_imputation['1'] = pd.DataFrame(ff_shares_spec).mean(axis=0)
    
    # print(feature_imputation)
    feature_imputation['2'] = np.ones(feature_imputation.shape[0])-feature_imputation['0']-feature_imputation['1'] 
    # print(feature_imputation)
    print('X shape:',X.shape)
    # assert(np.isnan(X).any() == False)
    
    print('y shape:',y.shape)
    assert(np.isnan(y).any() == False)
    
    return X, y,imputation,feature_imputation,pos,y_pat             
        

    
def create_feature_window(df,median,df_demo,demo_median,variables,los,specs):
    """
    Samples feature vectors from the input dfs. 

    Parameters
    ----------
    df : pd.DataFrame
        df with data of inidividual patient until moment of sampling
    df_train: pd.DataFrame
        df containing training set. Imputed values are based on the training set. 
    df_demo: pd.DataFrame
        demograhics df to sample from.
    df_demo_train: pd.DataFrame
        demograhics df containing the training set. Imputed values are based on the training set. 
    n: int
        feature_window
    variables: np.array[str]
        Array of strings representing the names of the variables to be included in the model.
    idx: str
        patient ID
        
    Returns
    -------
    v: feature vector
    type : np.array
    """
    
    from datetime import datetime, timedelta
    
    v = list() #define empty feature vector
    
    n = specs['feature_window']
    med_imp = 0 # define median imputation counter
    ff = 0 # define feed_forward imputer counter
    
    med_spec = list()
    ff_spec = list()
    
    # ------ Add demographics  ---------
    
    for i in range(len(df_demo)):
        if np.isnan(df_demo[i]):
            v.extend(np.nan*np.ones(1))
            # v.extend(demo_median[i]*np.ones(1))
            med_imp +=1
            med_spec.append(1)
            ff_spec.append(0)
        else:
            
            v.extend(df_demo[i]*np.ones(1))
            med_spec.append(0)
            ff_spec.append(0)
    
    # -------- Add LOS ----------------
    if specs['time']:
        v.extend(np.ones(1)*los)
        
    # ------ Add Raw vairables (labs / vitals) ---------------
    count=0
    for item in variables: #loop over features
        
        temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
        
        if temp.shape[0] < 1: # If snippet contains none for this feature
            a = np.ones(n)*median[count]
            v.extend(a)
            med_imp += n
            med_spec.append(1)
            ff_spec.append(0)
            
        elif temp.shape[0] < n: # If snippet contains less than n values for feature, impute with most recent value
            a = np.concatenate((temp[:,3], 
                                np.ones(n-temp.shape[0])*temp[-1,3]), axis=None)
            v.extend(a)
            ff += (n-temp.shape[0])
            
            med_spec.append(0)
            ff_spec.append((n-temp.shape[0])/n)
            
        else: #if snippet contains n or more values, take n most recent values
            a = temp[-n:,3]
            v.extend(a)          
            
            med_spec.append(0)
            ff_spec.append(0)
        count+=1
        
    med_share = med_imp/len(v)
    ff_share = ff/len(v)
    
    # -------- Add info_missingness ----------------
    
    if specs['freq']: # variable frequency
        
        for item in variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            a = np.ones(1) * (temp.shape[0]/los)
            v.extend(a)
            
    if specs['inter']: # variable interval
        
        for item in variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
                
            # interal between current and previous measurement
            if temp.shape[0] > 1:
                a = np.ones(1)*np.round((temp[-1,2]-temp[-2,2]).total_seconds()/3600.0,1)
            # less than 2 samples available?, no interval possible
            else:
                a = np.ones(1)*np.nan
            v.extend(a)
            
    # ---------- Add time series data ---------
    
    # Diff current - previous
    if specs['diff']:
        for item in variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            
            # difference between current and previous measurement
            if temp.shape[0] >= 2:
                a = np.ones(1)*(np.abs(temp[-1,3]-temp[-2,3]))
            # less than 2 samples available?, no difference possible
            else:
                a = np.ones(1)*np.nan
            v.extend(a)
    
    # stats in X hour sliding window
    if specs['stats']:
        for item in ['SpO2','HR','BP','RR']: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            
            # difference between current and previous measurement
            if temp.shape[0] >= 2:
                t = temp[-1,2] - timedelta(hours=specs['sliding_window'])
                temp = temp[np.where(temp[:,2]>t)]
            
                # add max
                a = np.ones(1)*np.max(temp[:,3])
                v.extend(a)
                #add min
                a = np.ones(1)*np.min(temp[:,3])
                v.extend(a)
                #add mean
                a = np.ones(1)*np.mean(temp[:,3])
                v.extend(a)
                #add median
                a = np.ones(1)*np.median(temp[:,3])
                v.extend(a)
                # add std
                a = np.ones(1)*np.std(temp[:,3])
                v.extend(a)
                
            # less than 2 samples available?, no stats
            else:
                a = np.ones(5)*np.nan
                v.extend(a)
            
            if temp.shape[0] >= 3:
                t = temp[-1,2] - timedelta(hours=specs['sliding_window'])
                temp = temp[np.where(temp[:,2]>t)]
            
                # add diff_std
                a = np.ones(1)*np.std(np.diff(temp[:,3]))
            # less than 3 samples available?, no std of diffs
            else:
                a = np.ones(1)*np.nan
            v.extend(a)
            
    return v,med_share,ff_share,med_spec,ff_spec
    
def create_dynamics(df,features,los,specs):
    v = list() #define empty feature vector
    
    if specs['time']:
        v.append(los)
        
    for item in features: #loop over features
        temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
        
        if temp.shape[0] < 1: # If snippet contains none for this feature
            v.append(np.nan)
           
        else: #if snippet contains n or more values, take n most recent values
            a = temp[-1,3]
            v.append(a)          

    

    return v
    
def KNN_imputer(X_train,X_train_raw,X_val,X_val_raw,X_test,X_test_raw,specs):
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=specs['knn'])
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    imputer_raw = KNNImputer(n_neighbors=specs['knn'])
    imputer_raw.fit(X_train_raw)
    X_train_raw = imputer_raw.transform(X_train_raw)
    X_val_raw = imputer_raw.transform(X_val_raw)
    X_test_raw = imputer_raw.transform(X_test_raw)
    
    print('NaNs imputed with KNN Imputer')
    return X_train,X_train_raw,X_val,X_val_raw,X_test,X_test_raw,imputer,imputer_raw

    
def balancer(X,y,undersampling=True):
    print('balancer triggered')
    """
    Balences classes.

    Parameters
    ----------
    X: np.array
        matrix [N feature vectors x N variables] with feature vectors
    y: np.array
        label vector [N feature vectors x 1]
    undersampling: optional: bool
        if True, use random undersampling. If False, use random oversampling.
    
    Returns
    -------
    X_bal: np.array
        balanced matrix [N feature vectors x N variables] with feature vectors
    y_bal: np.array
        balanced label vector [N feature vectors x 1]
    type : np.array
    """
    import imblearn
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler

    if undersampling:
        undersample = RandomUnderSampler(sampling_strategy='majority')
        print('Rebalance classes by random undersampling')
        X_bal, y_bal = undersample.fit_resample(X, y)
    else:
        
        oversample = RandomOverSampler(sampling_strategy='minority')
        print('Rebalance classes by random oversampling')
        X_bal, y_bal = oversample.fit_resample(X, y)
    
    print('After balancing: \n shape X',X_bal.shape,'n pos',sum(y_bal), 'n neg', len(y_bal)-sum(y_bal))
    return X_bal, y_bal

def optimize_n_trees(X,y):
    print('modeling with Random Forest')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(oob_score=True, warm_start=True, class_weight = 'balanced',verbose=0)
    min_estimators = 1
    max_estimators = 200
    
    oobs = []
    xs = []
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        oobs.append(oob_error)
        xs.append(i)
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    xs = np.asarray(xs)
    oobs = np.asarray(oobs)    
    
    plt.plot(xs, oobs,'-b')
    idx = np.argmin(oobs)
    plt.plot(xs[idx],oobs[idx],'*r')
    
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig('Tree_optimization',dpi=300)
    
    
    
    
    
def train_model(X_train,y_train,X_test,y_test,model,n_trees = 1000,class_weight='balanced'):
    print('train_model triggered')
    
    """
    Balences classes.

    Parameters
    ----------
    X_train: np.array
        Train set featurematrix [N feature vectors x N variables] with feature vectors
    y_train: np.array
        Train set label vector [N feature vectors x 1]
    X_test: np.array
        Test set featurematrix [N feature vectors x N variables] with feature vectors
    y_test: np.array
        Test set label vector [N feature vectors x 1]
    model: str
        model type: "LR" or "RF"
    
    Returns
    -------
    clf_ret: object
        trained classifier to be returned
    train_auc: float
        Area under the curve for model's performance on the train set
    explainer: object
        explainer for Shapley values based on the trained classifier
    """
    
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    import shap
    import skopt
    from skopt import BayesSearchCV
    import random
    import xgboost as xgb
    import time as timer
    
    if model == 'NN':
        
        
        import tensorflow as tf
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision()])
        model.fit(X_train, y_train, epochs=150, batch_size=10)
        _, precision = model.evaluate(X_train, y_train)
        print('Preciion: %.2f' % (precision*100))
        clf_ret = model
        auc_ret,ap_ret,_,_,_,_,p_ret,r_ret,t_ret = evaluate_metrics(clf_ret, X_test,y_test,NN=True)
    else:
        
        if model == 'RF':
            
            print('modeling with Random Forest')
            # define search space
           
            max_features = ['auto', 'log2','sqrt'] # Number of features to consider at every split
            max_depth = [3,5,7,9,11] # Maximum number of levels in tree, make not to deep to prevent overfitting
            
            param_grid = {  'max_features': max_features,
                            'max_depth': max_depth
                           }
            
            clf = RandomForestClassifier(class_weight = class_weight,verbose=0,n_estimators=n_trees)
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=1, random_state=random.randint(0, 10000))
            search = BayesSearchCV(estimator=clf,scoring='average_precision',n_iter=25,search_spaces=param_grid, n_jobs=-1, cv=cv)   
            
        elif model == 'LR':
            
            param_grid = {'penalty':['l1', 'l2'],'C': [0.001,0.01,0.1,1,10,100,1000],
                           'solver': ['liblinear']}
            
            clf = LogisticRegression(max_iter=1000,class_weight = class_weight,verbose=0)
            
        elif model == 'XGB':
            print('modeling with XGB')
            clf = xgb.XGBClassifier(objective = "binary:logistic",n_estimators=n_trees,scale_pos_weight=class_weight[1],eval_metric="logloss")
            
            param_grid = {
                "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
                # "subsamples": [0.5, 0.6, 0.7, 0.8, 0.9],
                "learning_rate": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2], # default 0.1 
                "max_depth": [3,4], # default 3
                # "reg_alplha":[0.0, 0.005, 0.01, 0.05, 0.1],
                # "reg_lambdas":[0.8, 1, 1.5, 2, 4]
            }
            cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=1, random_state=random.randint(0, 10000))
            search = BayesSearchCV(estimator=clf,scoring='average_precision',n_iter=25,search_spaces=param_grid, n_jobs=-1, cv=cv)   
            
       
        startTime = timer.time()
        search.fit(X_train, y_train)
        executionTime = (timer.time() - startTime)
        print('Execution time',model,' for hyper optimization:',str(executionTime))
        # report the best result
        print('best Hyperparams after optiomization:',search.best_params_)
        
        
        startTime = timer.time()
        clf.fit(X_train, y_train)
        executionTime = (timer.time() - startTime)
        print('Execution time',model,' for fitting without hyper optimization:',str(executionTime))
        
        print('Performance on test set with unoptimized model:')
        base_auc,base_ap,_,_,_,_,p_base,r_base,t_base = evaluate_metrics(clf, X_test, y_test)
        
        clf_opt = search.best_estimator_
        print('Perfromance on test set with optimized model:')
        opt_auc,opt_ap,_,_,_,_,p_opt,r_opt,t_opt = evaluate_metrics(clf_opt, X_test,y_test)
        
        print('Improvement of {:0.2f}%.'.format( 100 * (opt_ap - base_ap) / opt_ap))
        
        #Pick the model with best performance on the test set
        if opt_ap > base_ap:
            clf_ret = clf_opt
            ap_ret = opt_ap
            p_ret = p_opt
            r_ret = r_opt
            auc_ret = opt_auc
            t_ret = t_opt
        else:
            clf_ret = clf
            ap_ret = base_ap
            p_ret = p_base
            r_ret = r_base
            auc_ret = base_auc
            t_ret = t_base
    
        
    explainer = shap.TreeExplainer(clf_ret)
      
    return clf_ret,auc_ret,explainer,p_ret,r_ret,ap_ret,t_ret

def predict(model, test_features):
    print('predict triggered')
    
    predictions = model.predict_proba(test_features)[:,1]
    
    return predictions

def CM(model, test_features, test_labels,t):
    from sklearn.metrics import confusion_matrix
    proba = model.predict_proba(test_features)[:,1]
    preds = (proba>t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
    print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
    
def evaluate_metrics(model, test_features, test_labels,plot=False,NN=False):
     
    """
    Calculates evaluation metrics

    Parameters
    ----------
    model: object
        Trained model
    test_features: np.array
        feature matrix for set to be evalutated [N feature vectors x M variables] 
    test_labels: np.array
        label vector for set to be evaluated [N feature vectors x 1] 
    
    Returns
    -------
    auc: float
        Area under the ROC curve 
    tn: int
        Number of true negatives
    fp: int
        numer of false positives
    fn: int
        number of false negatives
    tp: int
        number of true positives
    precision: np.array
        array with precisions for different threshold values
    recall: np.array
        array with recalls for different threshold values
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import brier_score_loss
   
    
    # Independend of threshold
    if NN:
        predictions = model.predict(test_features)
    else:
        predictions = model.predict_proba(test_features)[:,1]
        
    auc = roc_auc_score(test_labels, predictions)
    precision, recall, thresholds = precision_recall_curve(test_labels, predictions)

    #leave out last value for precision(=1) and recall (=0)
    precision = precision[:-1]
    recall = recall[:-1]
    
    assert(precision.shape == recall.shape == thresholds.shape)
    
    # Optimize Threshold by ensuring sens > 0.5
    betas = [2,3,4,5,6,7,8,9,10]
    betas_2 = []
    recalls = []
    for beta in betas:
        
        f_scores = (1+beta**2)*recall*precision/(recall+(beta**2*precision))    
        idx = np.argwhere(np.isnan(f_scores))
        f_scores = np.delete(f_scores, idx)
        thresholds = np.delete(thresholds, idx)
        t = thresholds[np.argmax(f_scores)]
        proba = model.predict_proba(test_features)[:,1]
        preds = (proba>t).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
        ap = average_precision_score(test_labels, proba)
        recall = np.round(tp/(tp+fn),2)
        precision = np.round(tp/(tp+fp),2)
        recalls.append(recall)
        
        if recall > 0.5:
            betas_2.append(beta)
    betas_2 = np.asarray(betas_2)
    
    if len(betas_2) < 2:
        print('Something Up with the recall')
        print('Betas:',betas)
        print('recalls:', recalls)
        beta = 3
    else:
        beta = np.min(betas_2)
        print('Best Beta value to ensure 0.5 sens:',beta)
    
    f_scores = (1+beta**2)*recall*precision/(recall+(beta**2*precision))    
    idx = np.argwhere(np.isnan(f_scores))
    f_scores = np.delete(f_scores, idx)
    thresholds = np.delete(thresholds, idx)
    t = thresholds[np.argmax(f_scores)]
    proba = model.predict_proba(test_features)[:,1]
    preds = (proba>t).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
    ap = average_precision_score(test_labels, proba)
    recall = np.round(tp/(tp+fn),2)
    precision = np.round(tp/(tp+fp),2)
    
    print('Average Precision:',ap)
    print('AUC:',auc)
    print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
    print('sens:',np.round(tp/(tp+fn),2),'spec:',np.round(tn/(tn+fp),2))
    print('Recall:',recall,'Pecision:',precision)
    print('Brier score:',brier_score_loss(test_labels, predictions))
    
    precision, recall, thresholds = precision_recall_curve(test_labels, predictions)
    return auc,ap,tn, fp, fn, tp,precision,recall,t
    

def plot_F_curve(f,t,t_opt,beta):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(t,f)
    # plt.xlim(0,0.5)
    # plt.ylim(0,1)
    plt.axvline(x=t_opt,color = 'r')
    plt.xlabel('Threshold')
    plt.ylabel('F'+str(beta)+'-score')
    plt.title('F'+str(beta)+'-score curve')
    plt.savefig('F_curve',dpi=200)
  

def plot_PR_curve(p,r,pn,rn,pna,rna):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(r,p)
    plt.plot(rn,pn)
    plt.plot(rna,pna)
    plt.legend(['Model','NEWS','No skill'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.savefig('results/PR_curve')
    
def plot_roc_curve(clf,X_val,y_val,clf_news,X_val_news,y_val_news):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    

    plot1 = metrics.plot_roc_curve(clf, X_val, y_val)
    plot = metrics.plot_roc_curve(clf_news, X_val_news, y_val_news,ax=plot1.ax_)
    plt.legend(['Model','NEWS'])
    plt.savefig('results/ROC_curve')


def plot_feature_importance(clf,n,model):
    import matplotlib.pyplot as plt
    
    if model == 'RF':
        importances = clf.feature_importances_
        
    elif model == 'LR':
        # print(np.abs(clf.coef_[0,:]))
        importances = np.abs(clf.coef_[0,:])
    # print(len(importances))
    # print(importances)
    summed_importances = list()
    summed_importances.append(importances[0])
    summed_importances.append(importances[1])
    
    lab_importances = importances[2:]
    # print(len(lab_importances))
    for i in np.arange(0,lab_importances.shape[0],n):
        summed_importances.append(np.sum(lab_importances[i:i+n])/n)
    
    summed_importances = np.array(summed_importances)

    
    return summed_importances



def stacked_barchart(names,df,name):
    
    # df = pd.DataFrame([v1,v2,v3])
    # print(df)    
    # #define chart parameters
    N = df.shape[0]
    barWidth = .5
    xloc = np.arange(N)
    
    cols = df.columns
    # #create stacked bar chart
    plt.figure()
    p1 = plt.bar(xloc, df[cols[0]], width=barWidth, color='springgreen')

    p2 = plt.bar(xloc, df[cols[1]], bottom=df[cols[0]], width=barWidth, color='coral')
    p3 = plt.bar(xloc, df[cols[2]], bottom=df[cols[1]]+df[cols[0]], width=barWidth, color='blue')
    
    # #add labels, title, tick marks, and legend
    plt.ylabel('Relative feature vector')
    # plt.xlabel('Data set')
    plt.title('Imputation distribution')
    plt.xticks(xloc, (names))
    plt.xticks(rotation=90)
    # plt.yticks(np.arange(0, 41, 5))
    plt.legend((p1[0], p2[0],p3[0]), ('Median imputation', 'Feed Forward imputation','No imputation'))
    plt.tight_layout()
    plt.savefig('results/'+name,dpi=300)
    # plt.show()
    return df


def isolate_class(df,label):
    
   
    ids = []
    for i in np.unique(df['ID']):
        snip = df[df['ID']==i]
        if snip['DEPARTMENT'].str.contains('IC').sum() > 0:
            ids.append(i)
    
    mask = df['ID'].isin(ids)
    
    
    if label == 'pos':
        df = df[mask]
    else:
        df = df[~mask]
    
    print(len(np.unique(df['ID'])),'patients ultimately to ICU')
    
    return df


def prepare_feature_vectors_plot(df,median,df_demo,demo_median,
                                 df_raw,median_raw,df_demo_raw,demo_median_raw,
                                 features,pid,specs,inc_start=0,label='pos'):
    print('prepare_feature_vectors_plot triggered')

    from datetime import datetime, timedelta
    
    '''
    Func to plot risk over time for individual patients:
        Inputs:
            df: Dataframe in shape of 'single timestamp' matrix
            pred_window: length of prediction window [h]
            gap: length of gap between prediction time and start prediction window [h]
            freq: frequency for predictions [1/day]
    '''
    pos = list() #create empty list for pos vectors
    raw = list()
    

    df = df[['ID','VARIABLE','TIME','VALUE','DEPARTMENT']]
    df = np.asarray(df)
    df_raw = df_raw[['ID','VARIABLE','TIME','VALUE','DEPARTMENT']]
    df_raw = np.asarray(df_raw)    
    
    df_demo = df_demo[['ID','AGE','BMI']]
    df_demo = np.asarray(df_demo)
    df_demo_raw = df_demo_raw[['ID','AGE','BMI']]
    df_demo_raw = np.asarray(df_demo_raw)
    

    
    idxs = np.unique(df[:,0])
    idx = idxs[pid] # assign unique patient ID

    patient = df[np.where(df[:,0]==idx)]
    patient = patient[patient[:,2].argsort()]
    demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0]
    
    patient_raw = df_raw[np.where(df_raw[:,0]==idx)]
    patient_raw = patient_raw[patient_raw[:,2].argsort()]
    demo_raw = df_demo_raw[np.where(df_demo_raw[:,0]==idx)][:,1:][0]
    
    
    
    if label == 'pos':
        t_event = patient[np.where(patient[:,4]=='IC')][0,2]
    else:
        t_event = patient[-1,2]
      
    if (t_event - patient[0,2]).total_seconds() / 3600.0 < specs['gap']: # cannot label patients with stay shorter than the gap
        print('Too short!')
    else:

        ts = []            
        t = t_event #- timedelta(hours=gap) #initialize t
        los = np.round((t_event - patient[0,2]).total_seconds() / 3600.0,0) #initiate day variable (start with total stay, extract day according to interval)
        window = (t_event-patient[0,2]).total_seconds() / 3600.0
        print('Stay window:',window,'hours')
        
        for i in range(int(window/specs['int_pos'])+inc_start): # Make timestamp array for every 4 hours , ADD + 1 to have samples from the start!
            ts.append(t)
            t = t - timedelta(hours=specs['int_pos'])
            
        # count_day = 1
        for t in ts: # Make feature vector for every timestamp in ts
            
            temp = patient[np.where(patient[:,2]<t)]
            temp_raw = patient_raw[np.where(patient_raw[:,2]<t)]
            
            v,med_share,ff_share,med_spec,ff_spec = create_feature_window(temp,median,demo,demo_median,features,los,specs)
            v,_,_,_,_ = create_feature_window(temp,median,demo,demo_median,features,los,specs)
            v_raw,_,_,_,_ = create_feature_window(temp_raw,median_raw,demo_raw,demo_median_raw,features,los,specs)
            
            pos.append(v)
            raw.append(v_raw)
        
            # if (count_day*interval)%24 == 0:
            los -= specs['int_pos']
            
            # count_day += 1
    
    pos=np.array([np.array(x) for x in pos])
    print('shape complete df',pos.shape)
    raw = np.array([np.array(x) for x in raw])
        
    print('shape raw df:',raw.shape)
    return pos,raw,ts,t_event

def subplot(dyn,ts,risks,news,features,pid,t_event,feature_impacts,label,t,dict_unit,specs):
    import math
    from datetime import timedelta
    from matplotlib import dates 
    import matplotlib
    import matplotlib.pyplot as plt

    rotation = 90
    fontsize = 10
    labelpad = 80
    n_charac = 5
    fontsize = 9
    
    max_risk = np.max(risks[1:]) # Leave first risk out for scaling (because it is often very high)
    risk_lim = max_risk*1.2
    if risk_lim > 1:
        risk_lim = 1
    y_lim = max_risk
    
    # Normalize risks
    # risks = risks/t
    
    # print(features)
    n_subplots = len(features)+1


            
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(n_subplots,1,1)
    # fig.subplots_adjust(t)
    # ax.set_title('ICU Risk')
    
    ax.set_ylabel('Risk')
    ax.set_xlim([ts[-1], t_event])
    ax.plot(ts, risks, 'b-x')
    
    ax.set_ylim(0, risk_lim)
    
    
    # plot NEWS
    # ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    # ax2.plot(ts, news,'r-x')
    # ax2.set_ylim([0, 16])
    # ax2.set_ylabel("NEWS")

    ax.plot([ts[-1],t_event],np.ones(2)*t,'--')

    ax.legend(('Risk','Threshold'),bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode=None, ncol=2)
    # ax2.legend(['NEWS'],bbox_to_anchor=(0, 1, 1, 0), loc="lower right", mode=None, ncol=1)
    
    if label == 'pos' :
        ax.axvspan(t_event - timedelta(hours=specs['pred_window']+specs['gap']), t_event, color='red', alpha=0.15, lw=0)
        
    count = 1
    
    
    for feature in features:
        
        ax = fig.add_subplot(n_subplots,1,count+1)        
        ax.plot(ts,dyn[features[count-1]],'k*')
        
        ax.set_xlim([ts[-1], t_event])
        
        # Add text for values of features:
        # for j, v in enumerate(dyn[features[count-1]]):
        #     if math.isnan(v) == False:
        #         ax.text(ts[j], v,str(v),fontsize=fontsize)
            
        ax.set_ylabel(features[count-1] + dict_unit[features[count-1]], rotation=rotation, fontsize=fontsize)
        ax.legend(([features[count-1]]),bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode=None, ncol=1)
        
        
        if label == 'pos':
            ax.axvspan(t_event - timedelta(hours=specs['pred_window']+specs['gap']), t_event, color='red', alpha=0.15, lw=0)
    
        if dyn[features[count-1]].dropna().shape[0] > 0:
            print(dyn[features[count-1]])
            ax.set_ylim(min(dyn[features[count-1]].dropna())*0.8, max(dyn[features[count-1]].dropna())*1.2)
        
        #make twin plot for feature impact
        ax2=ax.twinx()
        
        # make a plot with different y-axis using second axis object
        FI = feature_impacts[features[count-1]]
        ax2.plot(ts,FI,'m--x')
        ax2.plot([ts[-1],t_event],np.zeros(2)*t,'--k')
        ax2.legend((['Feature impact (SHAP)','Feature impact treshold']),bbox_to_anchor=(0, 1, 1, 0), loc="lower right", mode=None, ncol=1)
        
        
        # ax2.set_ylim(min(FI)*0.8, max(FI)*1.2)
        ax2.set_ylim([-y_lim, y_lim])
        ax2.set_ylabel("Feature Impact")
        
        # plt.xticks(ts, " ")                
        
        if count == 8:
            print('POC ready to be plotted')
            ax.set_xlabel('Time')
            plt.tight_layout()
            plt.savefig('results/POC_plot_'+str(pid),dpi=300)
            print('DONE')
            
            # plt.show()
                
        count += 1  
                            
    return plt
            


def NEWS(sat,HR,BP,RR,temp,oxy=True,AVPU =False):
    news = 0 #initalize news score
    
    # resp rate
    if RR <= 8 or RR >= 25:
        news+=3
    elif RR >= 9 and RR <= 11:
        news += 1
    elif RR >=21 and RR <= 24:
        news += 2
    
    # SpO2
    if sat <= 91:
        news+= 3
    elif sat == 92 or sat == 93:
        news+= 2
    elif sat == 94 or sat == 95:
        news+= 1
    
    #temp
    if temp <= 35:
        news+= 3
    elif (temp >= 35.1 and temp <= 36) or (temp >= 38.1 and temp <= 39):
        news+= 1
    elif temp >= 39.1:
        news += 2
    
    # Bp
    if BP <= 90 or BP >= 220:
        news += 3
    elif BP >= 91 and BP <= 100:
        news += 2
    elif BP >= 101 and BP <= 110:
        news+= 1
    
    # HR
    if HR <= 40 or HR >= 131:
        news+= 3
    elif (HR >= 41 and HR <= 50) or (HR >= 91 and HR <= 110):
        news+= 1
    elif HR >= 111 and HR <= 130:
        news+=2
        
    # oxygen
    if oxy:
        news+= 2
    if AVPU:
        news+= 3
    return news    
        
    
def build_news(X_train,X_val,n,n_demo=3):
    
    # first merge val and test set
    # X_train =  np.concatenate((X_train, X_test), axis=0)
    
    # EWS(RR,sat,temp,BP,HR,oxy=False,AVPU =False):
    news_train = []
    df = X_train
    for i in range(df.shape[0]):
        news_train.append(NEWS(df[i,n_demo+(n-1)],      #SpO2
                               df[i,1+n_demo+(n-1)],    #HR
                               df[i,2+n_demo+(n-1)],    #BP
                               df[i,3+n_demo+(n-1)],    #Resp
                               df[i,4+n_demo+(n-1)]))   #Temp
    df = X_val
    news_val = []
    for i in range(df.shape[0]):
        news_val.append(NEWS(df[i,n_demo+(n-1)],
                               df[i,1+n_demo+(n-1)],
                               df[i,2+n_demo+(n-1)],
                               df[i,3+n_demo+(n-1)],
                               df[i,4+n_demo+(n-1)]))
    
    news_train=np.array([np.array(x) for x in news_train])
    news_val=np.array([np.array(x) for x in news_val])
    
    news_train = np.expand_dims(news_train, axis=1)
    news_val = np.expand_dims(news_val, axis=1)
    print(news_train.shape,news_val.shape)
    return news_train,news_val
    
def results_news(X,y,model):  
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    thresholds = np.arange(-1,19,1)
    if model == 'threshold':
        recalls = []
        precisions = []
        fprs = []
        for i in thresholds:
            pred = (X[:,0] > i).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            recall =  tp / (tp + fn)
            precision =  tp / (tp + fp)
            fpr = fp/(fp+tn)
            recalls.append(recall)
            precisions.append(precision)
            fprs.append(fpr)
        thresholds  = list(thresholds)
        thresholds.append(20)
        thresholds = np.asarray(thresholds)
        recalls.append(0)
        precisions.append(1)
        fprs.append(0)
        recalls = np.array([np.array(x) for x in recalls])
        precisions = np.array([np.array(x) for x in precisions])
        fprs = np.array([np.array(x) for x in fprs])
        idx = np.argwhere(np.isnan(precisions))
        precisions = np.delete(precisions, idx)
        recalls = np.delete(recalls, idx)
        fprs = np.delete(fprs,idx)

    if model == 'RF':
        
        clf = RandomForestClassifier(class_weight = 'balanced',verbose=0)
        clf.fit(X, y)
    elif model == 'LR':
        
        clf = LogisticRegression(class_weight = 'balanced',verbose=0)
        clf.fit(X, y)
    
    
    
    return precisions,recalls,fprs,thresholds
    
def AP_manually(p,r):
    ap = 0
    for i in range(p.shape[0]-1):
        ap += (p[i]*(r[i]-r[i+1]))
    return ap

def make_total_features(features,specs):
    total_features = ['BMI','AGE','LOS']+list(features)
    if specs['freq']:
        new_features = []
        for i in features:
            new_features.append(i+str('_freq'))
        
        total_features.extend(new_features)
    if specs['inter']:
        new_features = []
        for i in features:
            new_features.append(i+str('_inter'))
        
        total_features.extend(new_features)
    if specs['diff']:
        new_features = []
        for i in features:
            new_features.append(i+str('_diff'))
        
        total_features.extend(new_features)
    if specs['stats']:
        stats = ['_max','_min','_mean','_median','_std','_diff_std']
        for stat in stats:
            new_features = []
            for i in ['SpO2','HR','BP','RR']:
                new_features.append(i+stat)
            
            total_features.extend(new_features)
    total_features = np.asarray(total_features)
    print(total_features.shape)
    return total_features    
