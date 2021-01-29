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
                'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
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
                    10,11,
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
                'DOSSIER_BEGINDATUM':str,
                'DOSSIER_EINDDATUM':str,
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
            new_features.append(i+str('_signed_diff'))
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

def importer_MAASSTAD(file,encoding,sep,header,specs):
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

    print('updated')
    print('importer MAASSTAD triggered')
    

    
    usecols = ['Patientnummer','Leeftijd','BMI','ICOpname','Opnamedatum','Ontslagdatum','Bestemming','Opnametype',
                'Ingangstijd','Eindtijd','Afnamedatum','Bepalingomschrijving_kort','Uitslag','Eenheid']
                
    type_lib = {
                'Patientnummer':'category',
                'Leeftijd':int,
                'BMI':float,
                # 'LEEFTIJD': int,
                'Behandelbeperking ICOpname':'category',
                'Opnamedatum': str,
                'Ontslagdatum':str,
                'Bestemming':'category',
                'Opnametype':'category',
                'Ingangstijd':str,
                'Eindtijd':str,
                'Afnamedatum':str,
                'Bepalingomschrijving_kort':'category',
                'Uitslag':str,
                'Eenheid':'category',
                # 'UITSLAGDATUM':str,
                # 'RESERVE':str,
                
                }
    
    data =  pd.read_excel(file,
                          sheet_name = 'Lab',
                        sep=sep,
                        encoding=encoding,
                          index_col=False,
                            header=header,
                            usecols = usecols,
                            dtype=type_lib,)
                       
    
    data = data[usecols]
    print('MAASSTAD LAB Data imported')
      
    print(data.columns)
    print(data.shape)
    new_cols = ['PATIENTNR','LEEFTIJD','BMI','NOICU','OPNAMEDATUM','ONTSLAGDATUM','BESTEMMING','OPNAMETYPE','DOSSIER_BEGINDATUM',
                    'DOSSIER_EINDDATUM','AFNAMEDATUM','DESC','UITSLAG','UNIT' ]
    print(len(new_cols))
    data.columns = new_cols
    
    print(data.head())
    print('UNIQUE IDs MAASSTAD LAB data:', len(np.unique(data['PATIENTNR'])))
    
    
    usecols = ['Patientnummer','Opnamedatum','Ontslagdatum','Bestemming','Opnametype',
                'Ingangstijd','Eindtijd','Afnamedatum','Label','Uitslag','Eenheid']
                
    type_lib = {
                'Patientnummer':'category',
                'Leeftijd':int,
                'BMI':float,
                # 'LEEFTIJD': int,
                'Behandelbeperking ICOpname':'category',
                'Opnamedatum': str,
                'Ontslagdatum':str,
                'Bestemming':'category',
                'Opnametype':'category',
                'Ingangstijd':str,
                'Eindtijd':str,
                'Afnamedatum':str,
                'Bepalingomschrijving_kort':'category',
                'Uitslag':str,
                'Eenheid':'category',
                # 'UITSLAGDATUM':str,
                # 'RESERVE':str,
                
                }
    
    data_vitals =  pd.read_excel(file,
                          sheet_name = 'Vitale functies',
                        sep=sep,
                        encoding=encoding,
                          index_col=False,
                            header=header,
                            usecols = usecols,
                            dtype=type_lib,)
                       
    print('MAASSTAD VIATLS Data imported')
    data_vitals = data_vitals[usecols]
    
    print(data_vitals.columns)
    print(data_vitals.shape)
    new_cols = ['PATIENTNR','OPNAMEDATUM','ONTSLAGDATUM','BESTEMMING','OPNAMETYPE','DOSSIER_BEGINDATUM',
                    'DOSSIER_EINDDATUM','AFNAMEDATUM','DESC','UITSLAG','UNIT' ]
    print(len(new_cols))
    data_vitals.columns = new_cols
    
    print(data_vitals.head())
    print('UNIQUE IDs MAASSTAD VITALS data:', len(np.unique(data_vitals['PATIENTNR'])))
    print(data_vitals.DESC.value_counts())
    
    data_tot = pd.concat([data,data_vitals],axis=0)
    
    print('total data shape:',data_tot.shape)
    # # ---- Make Unit dictionary ---------
    # units_vitals = list(['[%]','[bpm]','[mmHg]','[/min]','[°C]'])
    # units = []
    # all_units = list(data['DESC'].unique())
    
    # for i in all_units:
    #     snip = data[data['DESC']==i].reset_index(drop=True)['UNIT'].dropna()
    #     if snip.shape[0] > 0:
    #         units.append('[' + snip[0] + ']')
    #     else:
    #         units.append(' ')
    
    # dict_units = dict(zip(all_units, units))
    
    # all_units = np.asarray(all_units)
    
    # if specs['freq']:
    #     new_features = []
    #     for i in all_units:
    #         new_features.append(i+str('_freq'))
    #     dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
    #     new_features = []
    #     for i in ['SpO2','HR','BP','RR','Temp']:
    #         new_features.append(i+str('_freq'))
    #     dict_units.update(dict(zip(new_features, list(['[/h]'])*len(new_features))))    
        
            
    # if specs['inter']:
    #     new_features = []
    #     for i in all_units:
    #         new_features.append(i+str('_inter'))
    #     dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
    #     new_features = []
    #     for i in ['SpO2','HR','BP','RR','Temp']:
    #         new_features.append(i+str('_inter'))
    #     dict_units.update(dict(zip(new_features, list(['[hrs]'])*len(new_features))))    
        
    # if specs['diff']:
    #     new_features = []
    #     for i in all_units:
    #         new_features.append(i+str('_diff'))
    #     dict_units.update(dict(zip(new_features, units)))    
        
    #     new_features = []
    #     for i in ['SpO2','HR','BP','RR','Temp']:
    #         new_features.append(i+str('_diff'))
    #     dict_units.update(dict(zip(new_features, units_vitals)))    
        
    # if specs['stats']:
        
    #     stats = ['_max','_min','_mean','_median','_std','_diff_std']
    #     for stat in stats:
    #         new_features = []
    #         for i in ['SpO2','HR','BP','RR']:
    #             new_features.append(i+stat)
    #             dict_units.update(dict(zip(new_features, units_vitals[:-1])))    
    
    # # default_data.update({'item3': 3})
    
    # dict_units.update({'BMI':''})
    # dict_units.update({'AGE':'[yrs]'})
    # dict_units.update({'SpO2':'[%]'})
    # dict_units.update({'HR':'[bpm]'})
    # dict_units.update({'BP':'[mmHg]'})
    # dict_units.update({'RR':'[/min]'})
    # dict_units.update({'Temp':'[°C]'})
    # dict_units.update({'LOS':'[hrs]'})
    
    
    

    return data_tot

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


def cleaner_MAASSTAD(data,specs):
    
    mask = data.UITSLAG.isna()
    data=data[~mask]
    mask = data.UITSLAG.str.startswith('/')
    data = data[~mask]
    
    # Fix BP
    s1 = data.shape
    snip = data[data.DESC == 'NIBP'].copy()
    
    BP = data[data.DESC == 'NIBP']['UITSLAG']
    mask = data['DESC'] == 'NIBP'
    data = data[~mask]
    
    new_BP = []
    for s in BP.values:
       new_BP.append(s[:s.find('/')])
    new_BP = np.asarray(new_BP)
    snip.UITSLAG = snip
    
    data = pd.concat([data,snip],axis=0)
    
    assert s1 == data.shape




    # clean 'non informative' features --> non-numerics
    data['UITSLAG'] = pd.to_numeric(data['UITSLAG'], errors='coerce')
    print('shape before non numeric cleaning:',data.shape)
    data = data[data['UITSLAG'].notna()]
    print('shape after:',data.shape)
    # trasform UITSLAG in floats
    data['UITSLAG'].astype(float)     



    #Dates
    date_format='%Y-%m-%d %H:%M:%S'
    
    type_lib = {
            'PATIENTNR':str,
               'BMI':np.float16,
               'DESC':str,
               'NOICU':str
               }
    data = data.astype(type_lib)
    
    dates = [
        'OPNAMEDATUM','ONTSLAGDATUM',
        'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM', 'AFNAMEDATUM']
    for i in dates:
        print(i)
        # print(data[i].head())
        mask = data[i].str.startswith('29')
        mask = mask.fillna(False)
        data = data[~mask]
        
    for i in dates:
        data[i]= data[i].str[:19]
        data[i] = pd.to_datetime(data[i],format = date_format)
        
    print(data.BESTEMMING.value_counts())
   
    data.OPNAMETYPE = data.OPNAMETYPE.str.replace('Icopname','IC')
    print('Unique opnametype:',np.unique(data.OPNAMETYPE))
    mask = data['OPNAMETYPE'].isna()
    print('number of unknown OPNAMETYPE:',sum(mask))
    
    # clean Blood pressure (keep only systolic)
    # HEREEE
    
 
    
    
    # print('Most frequent variables: \n',data.DESC.value_counts()[40:80])
    
    print('----Feature selection-----')
    
    features = ['Kreatinine','Natrium','Hemoglobine','RDW','MCV','Kalium','Trombocyten','Leucocyten',
             'CRP','LD','ALAT','ASAT','Ferritine',"Lymfocyten absoluut",'Lymfocyten',"Basofielen granulocyten absoluut","Basofiele granulocyten"]
    
    features_vitals = ['SpO2','HR','NIBP','RR','Temp']
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    
    print(len(features),'features included')
    mask = data['DESC'].isin(features)
    data = data[mask]
    
    
    # Change some fetaure names to merg with EMC
    data.DESC = data.DESC.str.replace('NIBP','BP')
    data.DESC = data.DESC.str.replace('Leucocyten','Leukocyten')
    data.DESC = data.DESC.str.replace('ALAT','ALAT (GPT)')
    data.DESC = data.DESC.str.replace('ASAT','ASAT (GOT)')
    data.DESC = data.DESC.str.replace('Lymfocyten absoluut',"Lymfo's abs")
    data.DESC = data.DESC.str.replace('Basofielen granulocyten absoluut',"Baso's abs")
    data.DESC = data.DESC.str.replace('Basofiele granulocyten',"Baso's")
    
    # Update features
    features = ['Kreatinine','Natrium','Hemoglobine','RDW','MCV','Kalium','Trombocyten','Leukocyten',
             'CRP','LD','ALAT (GPT)','ASAT (GOT)','Ferritine',"Lymfo's abs",'Lymfocyten',"Baso's abs","Baso's"]
    
    features_vitals = ['SpO2','HR','NIBP','RR','Temp']
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    
    ids = np.unique(data['PATIENTNR']) # Unique IDs in dataset
    print('N patients left:',len(ids))
    
    # filter patients with No IC policy
    no_ic_data = data[data['NOICU']=='ja']
    ids = np.unique(no_ic_data.PATIENTNR)
    print(len(ids),' patients with No ICU policy')
    mask = data.PATIENTNR.isin(ids)
    data  = data[~mask]
    
    col_list = ['PATIENTNR','BMI','LEEFTIJD',
            'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','OPNAMEDATUM','ONTSLAGDATUM','DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',]
    data = data[col_list]

    data.columns = ['ID','BMI','AGE',
            'DEPARTMENT','TIME','VARIABLE','VALUE','ADMISSION','DISCHARGE','START','END']
    
    
    ids = np.unique(data['ID']) # Unique IDs in dataset
    print('N patients left:',len(ids))
    return data,features

def cleaner_labs(data):
    print('cleaner labs triggered')
    data.info()
    

    #BMI
    data['BMI'] = data['BMI'].apply(lambda x: x.replace(',','.'))
    data['BMI'] = data['BMI'].astype(float)
    #Dates
    date_format='%Y-%m-%d %H:%M:%S'
    

    dates = [
        'OPNAMEDATUM','ONTSLAGDATUM',
        'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
        # 'UITSLAGDATUM',
             'AFNAMEDATUM']
    
    
    for i in dates:
        print(i)
        mask = data[i].str.startswith('29')
        data = data[~mask]
        
        data[i]= data[i].str[:19]
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
    data['PATIENTNR'] = data['PATIENTNR'].astype(float)
    
    
    # clean 'non informative' features --> non-numerics
    data['UITSLAG'] = pd.to_numeric(data['UITSLAG'], errors='coerce')
    print(data.shape)
    data = data[data['UITSLAG'].notna()]
    print(data.shape)
    # trasform UITSLAG in floats
    data['UITSLAG'].astype(float)       
    data['UITSLAG'].astype(int)
    
    # labels
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('PUK','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Dialyse','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Dagverpleging','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Anders klinisch','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Observatie','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Gastverblijf','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Afwezigheid','Klinische opname')
    data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('Verkeerd bed','Klinische opname')
    
    
   
    
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
    data['PATIENTNR'] = data['PATIENTNR'].astype(float)
    data['UITSLAG'].astype(int)
    
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
            'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','OPNAMEDATUM','ONTSLAGDATUM','DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',]
    df = df[col_list]

    df.columns = ['ID','BMI','AGE','DEST',
            'DEPARTMENT','TIME','VARIABLE','VALUE','ADMISSION','DISCHARGE','START','END']
    
    
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
        
        mask = df['ID'].isin(ids)
        df = df[mask]
        print('n patients left after filtering no ICU policy:', len(df['ID'].unique()))
    
    print(df.shape)
    
    # mask = df[df['DEPARTMENT']=='IC']['START'].min()

    
    print('Unique features:', df['VARIABLE'].unique())
    print('Unique patients:', len(df['ID'].unique()))
    
    # Set PIDs to strings
    df['ID'] = df['ID'].astype(str)
    
    return df,features




def get_ids(df):
    print('get_ids triggered')
    # get IDs of all ICU containing patients
    ids_IC = []
    for i in df['ID'].unique():
        opnametypes = df[df['ID']==i].sort_values(by='TIME').DEPARTMENT # filter out types of individual patient
        
        
        if len(opnametypes) == 0:
            print('Patient ', i, ' contains no type info')
        if opnametypes.str.contains('IC').sum() > 0:    # check if it contains any ICU admission
            ids_IC.append(i)
    
    # get idx of IC which also seen the clinic
    ids_events = []
    for i in ids_IC:
        opnametypes = df[df['ID']==i].sort_values(by='TIME')
        opnametypes = opnametypes.DEPARTMENT.dropna().reset_index(drop=True) # filter out types of individual patient
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


def fix_episodes(df):
    print('update')
    
    
    print(df.shape)

    A = pd.DataFrame()
    A['ID'] = np.unique(df.ID)
    a = list()
    d = list()
    for i in np.unique(df.ID):
        ex = df[df['ID']==i]
        a.append(len(np.unique(ex['ADMISSION'].dropna())))
        d.append(len(np.unique(ex['DISCHARGE'].dropna())))
    A['A'] = a
    A['D'] = d
    
    print(sum(A.A > 1),'episodial in',A.shape[0] ,'patients')
    
    ids = A.ID[A.A>1].values # get IDs wiht multiple episodes
    df_new = df[~df['ID'].isin(ids)] # Define new df without these
    
        
    for i in ids:
        ex = df[df['ID']==i]
        ex = ex.sort_values(by='TIME')
        admss = np.unique(ex.ADMISSION.dropna())
        diss = np.unique(ex.DISCHARGE.dropna())
        count = 0
    
    if len(admss) != len(diss):
        for i in range(len(admss)-1):
            
            snip = ex[(ex.TIME >= admss[i])&(ex.TIME <= diss[i])].reset_index(drop=True)
            if snip.shape[0] > 0:
                snip.loc[:,'ID'] = snip.loc[0,'ID'] + '_' + str(count)
                
                df_new = pd.concat([df_new,snip])
            
            count+=1
    else:
        for i in range(len(admss)):
                
                snip = ex[(ex.TIME >= admss[i])&(ex.TIME <= diss[i])].reset_index(drop=True)
                if snip.shape[0] > 0:
                    snip.loc[:,'ID'] = snip.loc[0,'ID'] + '_' + str(count)
                    
                    df_new = pd.concat([df_new,snip])
                
                count+=1
    df_new = df_new.reset_index(drop=True)
    print(df_new.shape, '(data without admission date is filtered')
    
    
    ids_event = []
    ids_ICU_only = []
    
    for i in np.unique(df_new['ID']):
        ex = df_new[df_new['ID']==i].sort_values(by='TIME').reset_index(drop=True)
        if ex['DEPARTMENT'][0] == 'IC':
            ids_ICU_only.append(i)
        elif (sum(ex.DEPARTMENT.str.contains('IC'))>0) & (ex.DEPARTMENT[0] == 'Klinische opname'):
            ids_event.append(i)
    
    print(len(ids_event),' pos episodes')
    print(len(ids_ICU_only), 'episodes who start at ICU')
    print(len(np.unique(df_new.ID)),' episodes total')
    df_new = df_new[~df_new.ID.isin(ids_ICU_only)] # Filter only IC patients
    
    
 
    return df_new,ids_event


    
    
    

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
 
def Demographics(df):
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
    
    return df_demo

def Split(X,y,y_pat,y_t,ids_events,random_state,specs):
    
    
    from sklearn.model_selection import train_test_split
    
   
    ids = np.unique(X[:,-1])
    
    
    print(' TOTAL ids:', len(ids))
    # Split raw df in training and validation set on patient level:
    
    ids_train,ids_val = train_test_split(ids, test_size=specs['val_share'],random_state=random_state,
                                         stratify=np.in1d(ids,ids_events))
   
    
    
    X_train_full = X[np.nonzero(ids_train[:,None] == X[:,-1])[1],:]
    y_train_full = y[np.nonzero(ids_train[:,None] == X[:,-1])[1]]
    
    X_val = X[np.nonzero(ids_val[:,None] == X[:,-1])[1],:]
    y_val = y[np.nonzero(ids_val[:,None] == X[:,-1])[1]]
    y_pat_val = y_pat[np.nonzero(ids_val[:,None] == X[:,-1])[1]]
    y_t_val = y_t[np.nonzero(ids_val[:,None] == X[:,-1])[1]]
    
    #split training set in training and testing set
    ids_train,ids_test = train_test_split(ids_train, test_size=specs['test_share'],random_state=random_state,
                                          stratify=np.in1d(ids_train,ids_events))
    
    X_train = X_train_full[np.nonzero(ids_train[:,None] == X_train_full[:,-1])[1],:]
    y_train = y_train_full[np.nonzero(ids_train[:,None] == X_train_full[:,-1])[1]]
        
    X_test = X_train_full[np.nonzero(ids_test[:,None] == X_train_full[:,-1])[1],:]
    y_test = y_train_full[np.nonzero(ids_test[:,None] == X_train_full[:,-1])[1]]
    
    # Remove Patient IDs from X
    X_train = X_train[:,:-1]
    X_val = X_val[:,:-1]
    X_test = X_test[:,:-1]
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
        
    
    print('Split train, val en test set: \n original shape: ',X.shape,
          '\n train shape: ',X_train.shape, 'unique patients: ', len(y_train),'positives: ',sum(y_train),
          '\n Val shape: ',X_val.shape, 'unique patient: ', len(y_val),'positives: ',sum(y_val),
          '\n Test shape: ',X_test.shape, 'unique patients: ', len(y_test),'positives: ',sum(y_test)
          )
    
    return X_train,y_train,X_val,y_val,y_pat_val,y_t_val,X_test,y_test


def Normalize(X_train,X_val,X_test,specs):
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    print('data normalized using standardscaler')
    
    
    return X_train_norm,X_val_norm,X_test_norm



def prepare_feature_vectors(df,df_demo,ids_events,features,specs):
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
    import numpy as np    
    

    pos = list() #create empty list for pos labeled feature vectors
    neg = list() #create empty list for neg labeled feature vectors
    y_pat = list()
    y_t = list()
    
    entry_dens_full = list()
    
        
    
    
    df_neg = df[~df['ID'].isin(ids_events)] # Isolate negative df --> no ICU 
    df_pos = df[df['ID'].isin(ids_events)]
    

    #Make arrays out of dfs, keep only ID, VARIABLE, TIME, VALUE, DEPARTMENT
    df_pos = df_pos[['ID','VARIABLE','TIME','VALUE','DEPARTMENT','DISCHARGE','START']]
    df_neg = df_neg[['ID','VARIABLE','TIME','VALUE','DEPARTMENT','DISCHARGE','START']]
    
    print('pos df:',df_pos.shape, '-->',len(df_pos['ID'].unique()), 'patients')
    print('neg df:',df_neg.shape, '-->',len(df_neg['ID'].unique()), 'patients')
    
    df_pos = np.asarray(df_pos)
    df_neg = np.asarray(df_neg)
    df_demo = np.asarray(df_demo)

    count = 0
    
    print('-----Sampling for positive patients-----') 
    
    n_pos_samples = int(specs['pred_window']/specs['int_pos'])
    print('max',n_pos_samples+1, 'positive samples per positive patient')
    
    for idx in np.unique(df_pos[:,0]):                      # loop over patients
        entry_dens_patient = list()
    
        patient = df_pos[np.where(df_pos[:,0]==idx)] # isolate pat id
        patient = patient[patient[:,2].argsort()]   # sort by date
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0] # isolate pat id
        
        t_event = patient[np.where(patient[:,4]=='IC')][:,6].min() # define moment of ICU admission 
        total_los = np.round((t_event - patient[:,2].min()).total_seconds() / 3600.0,0) #initialize LOS variable [hours] 
  
        if total_los <= 0: # cannot label patients for which time between start and event is shorter than the gap
            count += 1
            print('los shorter than zero:',total_los,'pos episode:',idx)
            
        else:

            if total_los <= specs['int_pos']:
                print('pos patient LOS shorter than interval')
                t = patient[:,2].min() + timedelta(hours=int(total_los/2))
                los = int(total_los/2)
                temp = patient[np.where(patient[:,2]<t)]
                if (los<0) | (los> total_los):
                    print('wrong: negative or too big los',los)
                v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx)
                t_to_event = int((t_event - t).total_seconds() / 3600.0)
                pos.append(v)
                y_pat.append(1) # add patient label
                y_t.append(t_to_event)
                entry_dens_patient.append(entry_dens)
                
            else:
                los = specs['int_pos'] #initialize los
                t = patient[:,2].min() # initialize t
                while los < total_los:
                    t = t + timedelta(hours=specs['int_pos'])
                    
                    
                    temp = patient[np.where(patient[:,2]<t)]
                    # ##### IF ENTRY DENISTY
                    # t_2 = t - timedelta(hours=24)
                    # temp = temp[np.where(temp[:,2]>=t_2)]
                    # ######
                    if (los<0) | (los> total_los):
                        print('wrong: negative or too big los',los)
                
                    v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx)
                    t_to_event = int((t_event - t).total_seconds() / 3600.0)
                
                    if t_to_event > (specs['pred_window']+specs['gap']):
                        neg.append(v) # add feature vector to 'neg' list
                    elif (t_to_event <= (specs['pred_window']+specs['gap'])) & (t_to_event > specs['gap']):
                        pos.append(v) # add feature vector to 'pos' list
                    else:
                        neg.append(v) # add feature vector to 'neg' list
                        
                    y_pat.append(1) # add patient label
                    y_t.append(t_to_event)
                    entry_dens_patient.append(entry_dens)
                        
                    los += specs['int_pos']
         
        if np.array(entry_dens_patient).shape[0] > 0:
            entry_dens_patient = list(np.array(entry_dens_patient).mean(axis=0))
            entry_dens_full.append(entry_dens_patient)
            
            
    print('number of pos patients with shorter stay than the defined GAP: ', count)              
    print('-----Sampling for negative patient-----')
    count=0
    for idx in np.unique(df_neg[:,0]): # loop over patients
        
        entry_dens_patient = list()
        
        patient = df_neg[np.where(df_neg[:,0]==idx)] # isolate by pid
        patient = patient[patient[:,2].argsort()]   # sort by date
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0]  # isolate by pid  
        
        
        if  pd.isnull(patient[:,2].min()):
            print('no times available')
        if pd.isnull(patient[:,5].min()):
            # print('no discharge available')
            t_event = patient[:,2].max()
        else:            
            t_event = patient[:,5].min() # event = discharge time
        
        
        
        los_total = np.round((t_event - patient[:,2].min()).total_seconds() / 3600.0,0) #initialize LOS variable [hours] 
            
        if total_los <= specs['gap']: # cannot label patients for which time between start and event is shorter than the gap
            count += 1
            print('los shorter than gap:',total_los,'neg episode:',idx)
            
        else:
            los =  specs['int_neg'] #initialize los
            t = patient[:,2].min() # initialize t
            while los < total_los:
                t = t + timedelta(hours=specs['int_neg'])
                
                temp = patient[np.where(patient[:,2]<t)]
                # ##### IF ENTRY DENISTY
                # t_2 = t - timedelta(hours=24)
                # temp = temp[np.where(temp[:,2]>=t_2)]
                # ######
                if (los<0) | (los> total_los):
                    print('wrong: negative or too big los',los)
                
                v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx)
                neg.append(v) # add feature vector to 'neg' list
                y_pat.append(0) # add patient label
                y_t.append(int((t_event - t).total_seconds() / 3600.0))
                entry_dens_patient.append(entry_dens)
                
                los += specs['int_neg']
                
        if np.array(entry_dens_patient).shape[0] > 0:
            entry_dens_patient = list(np.array(entry_dens_patient).mean(axis=0))
            entry_dens_full.append(entry_dens_patient)
            
    print('number of neg patients with shorter stay than the defined GAP: ', count)            
    print(len(pos),len(neg))
    pos=np.array([np.array(x) for x in pos])
    neg=np.array([np.array(x) for x in neg])
    y_pat = np.array([np.array(x) for x in y_pat])
    y_t = np.array([np.array(x) for x in y_t])
    
    print('shape of positive class: ', pos.shape, 'shape of negative class: ', neg.shape)
    
    X = np.concatenate((pos, neg), axis=0)
    y = np.concatenate((np.ones(pos.shape[0]),np.zeros(neg.shape[0])),axis=0)
    assert(y.shape == y_pat.shape)
    
    entry_dens_full = np.array(entry_dens_full)
    
    
    # print(feature_imputation)
    print('X shape:',X.shape)
    # assert(np.isnan(X).any() == False)
    
    print('y shape:',y.shape)
    assert(np.isnan(y).any() == False)
    
    return X, y,entry_dens_full,y_pat,y_t             
        

    
def create_feature_window(df,demo,variables,los,specs,idx):
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
    n = specs['feature_window'] # Feature window
    stat_features = ['SpO2','HR','BP','RR','Temp','po2_arterial','pco2_arterial','ph_arterial','pao2_over_fio2','base_excess']
    v_n = len(demo)+len(variables)
    
    entry_dens = list()
    
    
    # ------ Add demographics  ---------
    
    for i in range(len(demo)):
        if np.isnan(demo[i]):
            v.append(np.nan)
        else:
            v.append(demo[i])

    # -------- Add LOS ----------------
    if specs['time']:
        v_n += 1
        v.append(los)
        
    # ------ Add Raw vairables (labs / vitals) ---------------
    count=0
    for item in variables: #loop over features
        
        temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
        
        if temp.shape[0] < 1: # If snippet contains none for this feature
            v.append(np.nan)
            entry_dens.append(0)
            
        elif temp.shape[0] < n: # If snippet contains less than n values for feature, impute with most recent value
            v.extend(np.concatenate((temp[:,3],np.ones(n-temp.shape[0])*temp[-1,3]), axis=None))
            entry_dens.append(1)
            
        else: #if snippet contains n or more values, take n most recent values
            v.extend(temp[-n:,3])          
            entry_dens.append(1)
            
        count+=1
        
   
    
    # -------- Add info_missingness ----------------
    
       
    if specs['freq']: # variable frequency
        v_n += (len(variables)-5)
        
        info_miss_variables = list(variables)
        info_miss_variables.remove('BP')
        info_miss_variables.remove('RR')
        info_miss_variables.remove('HR')
        info_miss_variables.remove('Temp')
        info_miss_variables.remove('SpO2')
        info_miss_variables = np.asarray(info_miss_variables)
        
        for item in info_miss_variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            v.append(temp.shape[0]/los)
            entry_dens.append(1)
            
    if specs['inter']: # variable interval
        v_n += (len(variables)-5)
        for item in info_miss_variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
                
            # interal between current and previous measurement
            if temp.shape[0] > 1:
                v.append(np.round((temp[-1,2]-temp[-2,2]).total_seconds()/3600.0,1))
                entry_dens.append(1)
            
            else: # less than 2 samples available?, no interval possible
                v.append(np.nan)
                entry_dens.append(0)
                
            
    # ---------- Add time series data ---------
    
    # Diff current - previous
    if specs['diff']:
        v_n += len(variables)
        for item in variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            
            # difference between current and previous measurement
            if temp.shape[0] >= 2:
                v.append(temp[-1,3]-temp[-2,3])
                entry_dens.append(1)
            
            else: # less than 2 samples available?, no difference possible
                v.append(np.nan)
                entry_dens.append(0)
            
    # stats in X hour sliding window
    if specs['stats']:
        
        stat_features_present = [x for x in list(variables) if x in stat_features]
        v_n += (len(stat_features_present)*6)
        for item in stat_features_present:
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
     
            # difference between current and previous measurement
            if temp.shape[0] >= 2:
                t = temp[-1,2] - timedelta(hours=specs['sliding_window'])
                temp = temp[np.where(temp[:,2]>t)]
                
                # add max
                v.append(np.max(temp[:,3]))
                #add min
                v.append(np.min(temp[:,3]))
                #add mean
                v.append(np.mean(temp[:,3]))
                #add median
                v.append(np.median(temp[:,3]))
                # add std
                v.append(np.std(temp[:,3]))
                
                entry_dens = entry_dens + list([1,1,1,1,1])
                
            # less than 2 samples available?, no stats
            else:
                v.extend(np.ones(5)*np.nan)
                entry_dens = entry_dens + list([0,0,0,0,0])
                
            if temp.shape[0] >= 3:
                v.append(np.std(np.diff(temp[:,3]))) # add diff_std
                entry_dens.append(1)
            
            else: # less than 3 samples available?, no std of diffs
                v.append(np.nan)
                entry_dens.append(0)

    assert v_n == len(v)
    v.append(idx)
    
    return v,entry_dens

    
def Imputer(X_train,X_val,X_test,specs):
    
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=specs['knn'])
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    print('NaNs imputed with KNN Imputer')
    return X_train,X_val,X_test,imputer

    
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
          
       
       elif model == 'SVM':
           print('modeling with support vector machine')
           from sklearn import svm
           Cs = [0.001, 0.01, 0.1, 1, 10]
           gammas = [0.001, 0.01, 0.1, 1]
           param_grid = {'C': Cs, 'gamma' : gammas}
           clf = svm.SVC(kernel='rbf',probability=True)
           
       
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
       base_auc,base_ap = evaluate_metrics(clf, X_test, y_test,X_train,y_train)
       
       clf_opt = search.best_estimator_
       print('Perfromance on test set with optimized model:')
       opt_auc,opt_ap = evaluate_metrics(clf_opt, X_test,y_test,X_train,y_train)
       
       print('Improvement of {:0.2f}%.'.format( 100 * (opt_ap - base_ap) / opt_ap))
       
       #Pick the model with best performance on the test set
       if opt_ap > base_ap:
           clf_ret = clf_opt
           ap_ret = opt_ap
           auc_ret = opt_auc
       else:
           clf_ret = clf
           ap_ret = base_ap
           auc_ret = base_auc
           
   if (model == 'RF') or (model == 'XGB'):    
       explainer = shap.TreeExplainer(clf_ret)
   elif model == 'SVM':
       explainer = shap.KernelExplainer(clf_ret.predict,X_train)
     
   return clf_ret,explainer,auc_ret,ap_ret

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
    
def evaluate_metrics(model, test_features, test_labels,train_features, train_labels,plot=False,NN=False):
     
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
   
    
    
    print('Train set:')
    # Independend of threshold
    if NN:
        predictions = model.predict(train_features)
    else:
        predictions = model.predict_proba(train_features)[:,1]
        
    auc = roc_auc_score(train_labels, predictions)
    ap = average_precision_score(train_labels, predictions)
    print('AUC: ',auc, ' AP:', ap)
    
    print('Test set:')
    # Independend of threshold
    if NN:
        predictions = model.predict(test_features)
    else:
        predictions = model.predict_proba(test_features)[:,1]
        
    auc = roc_auc_score(test_labels, predictions)
    ap = average_precision_score(test_labels, predictions)
    print('AUC: ',auc, ' AP:', ap)
    
    
    return auc,ap
    

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

def make_total_features(features,specs,demo=True):
    
    if demo:
        total_features = ['BMI','AGE','LOS']+list(features)
    else:
        total_features = list(features)
        
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
            new_features.append(i+str('_signed_diff'))
        
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

def Utility_score(y_pred,y_true,y_t,specs):
    
    """
    Make Utilty score
    """
    U = 0 # TN
    
    U_TP = np.concatenate([np.linspace(0.1,1,specs['pred_window']-1),np.linspace(1,0.5,specs['gap'])],axis=0)
    U_FN = np.linspace(0,-2,specs['pred_window']+specs['gap'])
    
    if y_true == 0:
        if y_pred == 1: # FP
            U = -0.1
    
    if y_true == 1:
        if y_pred == 1: # TP
            U = U_TP[-y_t]
        elif y_pred == 0: # FN
            U = U_FN[-y_t]
    return U
    
def make_table(df):
    ex = list()
    ex.append(str(np.round(np.mean(df.AGE),1))+' ('+str(np.round(np.std(df.AGE),1))+')')
    ex.append(str(np.round(np.median(df.dropna().AGE),1))+' ('+str(np.round(np.min(df.dropna().AGE),1))+','+str(np.round(np.max(df.dropna().AGE),1)) +')')
    n = sum((df.AGE<18))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>=18) & (df.AGE<=45))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>45) & (df.AGE<=65))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>65) & (df.AGE<=80))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.AGE>80))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum(df.AGE.isna())
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    
    ex.append(str(np.round(np.mean(df.dropna().BMI),1))+' ('+str(np.round(np.std(df.dropna().BMI),1))+')')
    ex.append(str(np.round(np.median(df.dropna().BMI),1))+' ('+str(np.round(np.min(df.dropna().BMI),1))+','+str(np.round(np.max(df.dropna().BMI),1)) +')')
    
    n = sum((df.LOS<=24))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>24) & (df.LOS<=72))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>72) & (df.LOS<=240))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>240))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
        
    return ex