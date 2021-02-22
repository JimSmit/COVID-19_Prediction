# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:23:05 2020

@author: Jim Smit

Functions


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
    """ 
    Function to impuate file with No-ICU policy information
    
    """ 
    df = pd.read_excel(file, header=0)
    
    type_lib = {
                'ID':float,
                'LIM': float,
                }
    df = df.astype(type_lib)
    
    return df


    
def importer_labs(file,encoding,sep,header,specs,labs=True,filter=True,nrows=None,skiprows=None,skipinitialspace=False):
    """
    Import labs data.

    Parameters
    ----------
    file : [str]
        `hash_patient_id`s of patients to query.
    columns : Optional[List[str]]
        List of columns to return.

    Returns
    -------
    data       type : pd.DataFrame
    dict_units type : dict
    
    """


    print('importer labs triggered')
    
    col_list = ['PATIENTNR','SEX','OMSCHRIJVING','BMI','LEEFTIJD',
                'OPNAMEDATUM','ONTSLAGDATUM','HERKOMST',
                'BESTEMMING',
            'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
            'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','UNIT'
            ]

    if filter:
        usecols = [0,1,2,3,
                    4,5,6,8,
                   10,11,12,
                   14,15,18,19,20]
    else:
        usecols = None
    
    type_lib = {
                'SUBJECTNR':str,
                'OMSCHRIJVING':'category',
                'GESLACHT':str,
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
                'EENHEID':str,
                # 'UITSLAGDATUM':str,
                # 'RESERVE':str,
                
                }
 
    data =  pd.read_excel(file,
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
    print(data.columns)
    # print()

    # Make DF with unique labs and corresponding units
    

    
    
    col_list = ['SUBJECTNR','GESLACHT','OMSCHRIJVING','BMI','LEEFTIJD',
                    'OPNAMEDATUM','ONTSLAGDATUM','HERKOMST',
                    'BESTEMMING',
                'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
                'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','EENHEID'
                ]
    data = data[col_list]
    
    data = data.astype(type_lib)
    data.columns = ['PATIENTNR','SEX','OMSCHRIJVING','BMI','LEEFTIJD',
                    'OPNAMEDATUM','ONTSLAGDATUM','HERKOMST',
                    'BESTEMMING',
                'DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM',
                'OPNAMETYPE','AFNAMEDATUM','DESC','UITSLAG','UNIT'
                ]
     
    
    
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


def importer_vitals(file,encoding,sep,header,):
    print('importer vitals triggered')
    extract = [0,2,6,8,9,10,11,12,14]
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
    
    data =  pd.read_excel(file,
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


def importer_MAASSTAD(file,encoding,sep,header,specs):
    """
    Import MAASSTAD data.

    Returns
    -------
    data_tot     type : pd.DataFrame
    data_vitals    type : pd.DataFrame
    
    """

    print('importer MAASSTAD triggered')
    

    
    usecols = ['Patientnummer','Leeftijd','Geslacht','BMI','ICOpname','Opnamedatum','Ontslagdatum','Bestemming','Opnametype',
                'Ingangstijd','Eindtijd','Afnamedatum','Bepalingomschrijving_kort','Uitslag','Eenheid']
                
    type_lib = {
                'Patientnummer':'category',
                'Leeftijd':int,
                'BMI':float,
                'Geslacht':str,
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
    new_cols = ['PATIENTNR','LEEFTIJD','SEX','BMI','NOICU','OPNAMEDATUM','ONTSLAGDATUM','BESTEMMING','OPNAMETYPE','DOSSIER_BEGINDATUM',
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

    
    return data_tot,data_vitals




def cleaner_MAASSTAD(data,specs):
    
    print(len(data.PATIENTNR.unique()),' unique patients at start')
    
    #SEX
    data.loc[data['SEX']=='M','SEX'] = 1
    data.loc[data['SEX']=='V','SEX'] = 0
    print('unique sexes in dataset:',data.SEX.unique())
    
    mask = data.UITSLAG.isna()
    data=data[~mask]
    mask = data.UITSLAG.str.startswith('/')
    data = data[~mask]
    
    # clean Blood pressure (keep only systolic)
    s1 = data.shape
    snip = data[data.DESC == 'NIBP'].copy()
    
    BP = data[data.DESC == 'NIBP']['UITSLAG']
    mask = data['DESC'] == 'NIBP'
    data = data[~mask]
    
    new_BP = []
    for s in BP.values:
       new_BP.append(s[:s.find('/')])
    new_BP = np.asarray(new_BP)
    snip.UITSLAG = new_BP
    
    data = pd.concat([data,snip],axis=0)
    
    assert s1 == data.shape

    
    # clean 'non informative' features --> non-numerics
    data['UITSLAG'] = pd.to_numeric(data['UITSLAG'], errors='coerce')
    print('shape before non numeric cleaning:',data.shape)
    data = data[data['UITSLAG'].notna()]
    print('shape after:',data.shape)
    # trasform UITSLAG in floats
    data.loc[:,'UITSLAG'] = data['UITSLAG'].astype(float)     


    
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
        
   
    data.OPNAMETYPE = data.OPNAMETYPE.str.replace('Icopname','IC')
    print('Unique opnametype:',np.unique(data.OPNAMETYPE))
    mask = data['OPNAMETYPE'].isna()
    print('number of unknown OPNAMETYPE:',sum(mask))
    
    # print('Most frequent variables: \n',data.DESC.value_counts()[40:80])
    
    print('----Feature selection-----')
    
    features = ['Kreatinine','Natrium','Hemoglobine','RDW','MCV','Kalium','Trombocyten','Leucocyten',
             'CRP','LD','ALAT','ASAT','Ferritine',"Lymfocyten absoluut",'Lymfocyten',"Basofielen granulocyten absoluut","Basofiele granulocyten"]
    
    features_vitals = ['SpO2','HR','NIBP','Resp','Temp','FiO2']
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    
    
    mask = data['DESC'].isin(features)
    data = data[mask]
    
    
    # Change some fetaure names to merg with EMC
    data.DESC = data.DESC.str.replace('NIBP','BP')
    data.DESC = data.DESC.str.replace('Resp','RR')
    data.DESC = data.DESC.str.replace('Leucocyten','Leukocyten')
    data.DESC = data.DESC.str.replace('ALAT','ALAT (GPT)')
    data.DESC = data.DESC.str.replace('ASAT','ASAT (GOT)')
    data.DESC = data.DESC.str.replace('Lymfocyten absoluut',"Lymfo's abs")
    data.DESC = data.DESC.str.replace('Basofielen granulocyten absoluut',"Baso's abs")
    data.DESC = data.DESC.str.replace('Basofiele granulocyten',"Baso's")
    
    # Manual Feature selection
    features = [
        # 'Hemoglobine',
        'RDW',
        # 'MCV',
        # 'Leukocyten',
              'CRP','LD',
               'ALAT (GPT)','ASAT (GOT)','Ferritine'
              ]
    # features =  ['Kreatinine','Natrium','Hemoglobine','RDW','MCV','Kalium','Trombocyten','Leukocyten',
    #           'CRP','LD','ALAT (GPT)','ASAT (GOT)','Ferritine',"Lymfo's abs",'Lymfocyten',"Baso's abs","Baso's"]
    
    features_vitals = ['SpO2','HR','BP','RR','Temp','FiO2']
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    
    mask = data['DESC'].isin(features)
    data = data[mask]
    
    data['OPNAMETYPE_RAW'] = data.OPNAMETYPE
    
    
    ids = np.unique(data['PATIENTNR']) # Unique IDs in dataset
    print('N patients left:',len(ids))
    
    print(data.shape)
    # filter patients with No IC policy
    no_ic_data = data[data['NOICU']=='ja']
    ids = np.unique(no_ic_data.PATIENTNR)
    print(len(ids),' patients with No ICU policy')
    mask = data.PATIENTNR.isin(ids)
    data  = data[~mask]
    print(data.shape)
    
    col_list = ['PATIENTNR','BMI','LEEFTIJD','SEX',
            'OPNAMETYPE','OPNAMETYPE_RAW','AFNAMEDATUM','DESC','UITSLAG','OPNAMEDATUM','ONTSLAGDATUM','DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM','BESTEMMING']
    data = data[col_list]

    data.columns = ['ID','BMI','AGE','SEX',
            'DEPARTMENT','OPNAMETYPE_RAW','TIME','VARIABLE','VALUE','ADMISSION','DISCHARGE','START','END','DEST']
    
    
    ids = np.unique(data['ID']) # Unique IDs in dataset
    print('N patients left:',len(ids))
    
    print(data.VARIABLE.value_counts())
    return data,features

def cleaner_labs(data):
    print('cleaner labs triggered')
    data.info()
    
    #SEX
    data.loc[data['SEX']=='M','SEX'] = 1
    data.loc[data['SEX']=='V','SEX'] = 0
    print('unique sexes in dataset:',data.SEX.unique())
    
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
    data = data.astype({"UITSLAG": float})
    

    data['OPNAMETYPE_RAW'] = data.OPNAMETYPE
    
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
    print('N patients filtered due to unkonwn destination or department:', len(ids_unknown))        
            
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
    
    
    unique_features = np.unique(data.DESC)
    
    print('Lab Data cleaned')
    
    return df_lab,unique_features

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
    data = data.astype({"UITSLAG": float})    
       
    
    # filter columns with mean and diastolic BP
    data = data.drop(['Data2' , 'Data3'] , axis='columns')
    
    #clean PATIENTNR

    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('sub', '', regex=True)
    data.loc[:,'PATIENTNR'] = data['PATIENTNR'].str.replace('@', '', regex=True)
    
    mask = data['PATIENTNR'].str.contains('nan')
    data = data[~mask]
    data['PATIENTNR'] = data['PATIENTNR'].astype(float)

    
            
    # data['Data2'] = data['Data2'].astype(np.float16)
    # data['Data3'] = data['Data3'].astype(np.float16)
    data['AFNAMEDATUM'] = data['AFNAMEDATUM'].str[:19]
    data['AFNAMEDATUM'] = pd.to_datetime(data['AFNAMEDATUM'],format='%Y-%m-%d %H:%M:%S')
    
    data['DESC'] = data['DESC'].str.replace('ABP','BP')
    data['DESC'] = data['DESC'].str.replace('NIBP','BP')
    data['DESC'] = data['DESC'].str.replace('Resp(vent)','RespVent')
    data['DESC'] = data['DESC'].str.replace('Resp','RR')
    data['DESC'] = data['DESC'].str.replace('RRVent','RR')
    data.loc[data.DESC == 'FiO2(set)','DESC'] = 'FiO2'
    # data['DESC'] = data['DESC'].str.replace('FiO2(set)','FiO2')
    
    data['OPNAMETYPE_RAW'] = data.OPNAMETYPE
    
    print(data['OPNAMETYPE'].unique())
    print('UNIQUE OPNAMETYPE VITALS: \n', data['OPNAMETYPE'].value_counts())
    
    # data = data[data['OPNAMETYPE']=='USER']
    # data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('ONBEKEND','USER')
    
    # print(data.shape)
    # data['OPNAMETYPE'] = data['OPNAMETYPE'].str.replace('USER','Klinische opname')
    
    print('Unique types in vitals set:',data['OPNAMETYPE'].unique())
    df_vitals = data.reset_index(drop=True)
    print('Vitals Data cleaned')
    
    print(data[data.DESC=='FiO2(set)'])
    return df_vitals


def df_merger(df_1,df_2,df_cci,specs):
    print('df_merger triggered')
    
    print(df_1['OPNAMETYPE'].value_counts())
    print(df_2['OPNAMETYPE'].value_counts())
    
    print(df_1.columns)
    print(df_2.columns)
    df = pd.concat([df_1,df_2],axis=0)
    
    print(df.columns)
    
    print('----Feature selection-----')
    
    print('Vitals features ranking:')
    print(df_2['DESC'].value_counts())
    
    
    # Manual feature selection
    features = [
        # 'Hemoglobine',
        'RDW',
        # 'MCV',
        # 'Leukocyten',
             'CRP','LD',
              'ALAT (GPT)','ASAT (GOT)','Ferritine'
             ]
    # features =  ['Kreatinine','Natrium','Hemoglobine','RDW','MCV','Kalium','Trombocyten','Leukocyten',
    #           'CRP','LD','ALAT (GPT)','ASAT (GOT)','Ferritine',"Lymfo's abs",'Lymfocyten',"Baso's abs","Baso's"]
    
    # features_vitals = df_2['DESC'].value_counts().index
    # freqs = df_2['DESC'].value_counts().values
    # idx = np.where(freqs > 0.8*freqs[0])
    # features_vitals = features_vitals[idx]
    
    features_vitals = ['SpO2','HR','BP','RR','Temp','FiO2']
    print(len(features_vitals),' vitals features included')
    
    #merge features labs and vitals
    features = np.concatenate((features_vitals,features))
    # features = np.asarray(features_vitals)

    print('N features (Unique vitals + labs): ', features.shape)
    mask = df['DESC'].isin(features)
    df = df[mask]

    
    col_list = ['PATIENTNR','BMI','LEEFTIJD','SEX',
            'OPNAMETYPE','OPNAMETYPE_RAW','AFNAMEDATUM','DESC','UITSLAG','OPNAMEDATUM','ONTSLAGDATUM','DOSSIER_BEGINDATUM','DOSSIER_EINDDATUM','BESTEMMING']
    df = df[col_list]

    df.columns =  ['ID','BMI','AGE','SEX',
            'DEPARTMENT','OPNAMETYPE_RAW','TIME','VARIABLE','VALUE','ADMISSION','DISCHARGE','START','END','DEST']
   
    
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
    
    print(df.shape)
    if specs['policy']:
        print('Filter NO-IC policy patients')
        # Filter No-ICU policy patients
        
        pol_idx = df_cci[(df_cci['LIM'] == 1)|(df_cci['LIM'] == 5)]['ID'].values # also take doubts
        
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

    df = df.astype({"ID":str,"VALUE": float})  
    
    return df,features




def get_ids(df):  # identify patient IDs for 'event' group, only clinic group en ICU only group
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


def fix_episodes(df,specs):
    from datetime import datetime, timedelta

    print(df.shape)


    # ------ First fix multiple admissions ------------
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
    
    
    
    count_no_adm_date = A[A['A']==0].shape[0]
    ids_no_adm = A[A['A']==0].ID
    df = df[~df['ID'].isin(ids_no_adm)] # Define new df without these
    print(count_no_adm_date, 'patients filtered because no admission data available')
    
    count_no_dis_date = A[A['D']==0].shape[0]
    ids_no_dis = A[A['D']==0].ID
    df = df[~df['ID'].isin(ids_no_dis)] # Define new df without these
    print(count_no_dis_date, 'patients filtered because no discharge data available')
    
    print(df['ID'].unique().shape[0],' patients left')
    
    
    print(sum(A.A > 1),'multiple admissions')
    
    ids = A.ID[A.A>1].values # get IDs wiht multiple admission dates
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
    
    
    
    print(df_new.shape)
    print(len(np.unique(df_new.ID)),' episodes total')            
    
    # HERE FIND INDEXES WHO WERE ADMITTED TO ICU IMMEDIATELY
    ids_ICU_only = []
    
    for i in np.unique(df_new['ID']):
        ex = df_new[df_new['ID']==i].sort_values(by='TIME').reset_index(drop=True)
        
        if ex[ex['DEPARTMENT'] == 'IC'].shape[0] < 1:
            continue #No ICU admission found
        else:
            if ex[ex['DEPARTMENT'] == 'IC'].START.min() < ex.TIME.min():
                ids_ICU_only.append(i)
                
                
    
    print(len(ids_ICU_only), 'episodes who start at ICU (filtered)')
    df_new = df_new[~df_new.ID.isin(ids_ICU_only)] # Filter only IC patients
    print(len(np.unique(df_new.ID)),' episodes left')
    
    # HERE FIND INDEXES WHO MADE TRANSFER CLINIC --> ICU
    ids_event = []
    
    for i in df_new.ID.unique():
        ex = df_new[df_new['ID']==i].sort_values(by='TIME').reset_index(drop=True)
        if ex[ex['DEPARTMENT'] == 'IC'].shape[0] > 0:
            ids_event.append(i)
        
    print('of which ',len(ids_event),' pos episodes')
    
    # Fix episodes with multiple clinical episodes (ONLY KEEP MOST RECENT B4 ICU ADMISSION / DISCHARGE)
    # count = 0
    # ids = []
    # for i in df_new.ID.unique():
    #     ex = df_new[df_new['ID']==i].sort_values(by='TIME').reset_index(drop=True)
    #     if  ex[ex.DEPARTMENT != 'IC'].START.dropna().unique().shape[0] > 1:
    #         count += 1
    #         ids.append(i)
    
    # print(count, ' number of episodes with multiple clinical episodes')
    # df_new_2 = df_new[~df_new['ID'].isin(ids)] # Define new df without these
    
    # for i in ids:
    #     ex = df_new[df_new['ID']==i].sort_values(by='TIME').reset_index(drop=True)
    #     t = ex[ex.DEPARTMENT != 'IC'].START.dropna().max()
    #     snip = ex[ex.TIME > t]
    #     df_new_2 = pd.concat([df_new_2,snip])
        
    
    # # Fix Transfer episodes --> ONLY for non-event patients!
    # print('Bestemmingen:')
    # print(df_new_2.DEST.value_counts())
    df_new_2 = df_new
    
    a = list()
    transfer_pats = pd.DataFrame()
    
    for i in df_new_2.ID.unique():
        ex = df_new_2[df_new_2.ID == i].sort_values(by='TIME').reset_index(drop=True)
        if ((ex.DEST[0] == 'Ander ziekenhuis') | (ex.DEST[0] == 'Onbekend')| (ex.DEST[0] == 'Ziekenhuis buitenland')) & (i not in ids_event):
            a.append(i)
            ex_filtered = ex[ex['TIME']<(ex.TIME.max()-timedelta(hours=specs['pred_window']))]
            transfer_pats = pd.concat([transfer_pats,ex_filtered],axis=0)
    
    print(len(a),' Transfer tp other hospital patients identified')
    mask = df_new_2.ID.isin(a)
    
    df_new_2 = df_new_2[~mask] # remove transfer from bulk
    df_new_2 = pd.concat([df_new_2,transfer_pats],axis=0) #merge again with filtered 'transfer' patients
    
    
    df_new_2.columns = ['ID','BMI','AGE','SEX',
            'DEPARTMENT','OPNAMETYPE_RAW','TIME','VARIABLE','VALUE','ADMISSION','DISCHARGE','START','END','DEST']

 
    return df_new,ids_event


def Demographics(df,specs):  # create df with demographics for all IDs
    ids = np.unique(df['ID']) # Unique IDs in dataset
        
    # create df with Demographics data (BMI and AGE) -- >  df_demo
    bmi = []
    age =[]
    sex=[]
    for i in ids:
        temp = df[df['ID']==i]
        mask =  temp['BMI'].isna()
        if temp[~mask].shape[0]>0:
            bmi.append(temp[~mask].reset_index()['BMI'][0])
        else:
            bmi.append(np.nan)
        
        mask =  temp['AGE'].isna()
        if temp[~mask].shape[0]>0:
            age.append(float(temp[~mask].reset_index()['AGE'][0]))
        else:
            age.append(np.nan)
        mask =  temp['SEX'].isna()
        if temp[~mask].shape[0]>0:
            sex.append(float(temp[~mask].reset_index()['SEX'][0]))
        else:
            sex.append(np.nan)
    
    df_demo = pd.DataFrame()
    df_demo['ID'] = ids
    
    df_demo['BMI'] = bmi
    df_demo['SEX'] = sex
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
    
    # filter demogrohics to use
    print(specs['n_demo'])
    cols = ['ID'] + specs['n_demo']
    
    print(cols)
    df_demo = df_demo[cols]
    
    
    return df_demo


def Split(X,y,y_pat,y_t,ids_events,random_state,specs,ids_CV,random_split):
    
    
    from sklearn.model_selection import train_test_split
    
   
    ids = np.unique(X[:,-1])
    
    
    print(' TOTAL ids:', len(ids))
    # Split raw df in training and validation set on patient level:
    
        
    if random_split:
        ids_train,ids_val = train_test_split(ids, test_size=specs['val_share'],random_state=random_state,
                                             stratify=np.in1d(ids,ids_events))
   
    else:
        ids_val = ids_CV
                
        ids_train = np.array([x for x in list(ids) if x not in list(ids_val)])
    
    
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
    
    
    # assert no Ids are overlapping
    train = list(ids_train)
    test = list(ids_test)
    val = list(ids_val)
    assert len([x for x in test if x in train]) == 0
    assert len([x for x in val if x in train]) == 0
    
    
    # Remove Patient IDs from X
    X_train = X_train[:,:-1]
    X_val = X_val[:,:-1]
    X_test = X_test[:,:-1]
    
    #Transform to floats
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_test = X_test.astype(float)
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
        
    
    print('Split train, val en test set: \n original shape: ',X.shape,
          '\n train shape: ',X_train.shape, 'unique feature vectors: ', len(y_train),'positives: ',sum(y_train),
          '\n Val shape: ',X_val.shape, 'unique feature vectors: ', len(y_val),'positives: ',sum(y_val),
          '\n Test shape: ',X_test.shape, 'unique feature vectors: ', len(y_test),'positives: ',sum(y_test)
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
    
    
    return X_train_norm,X_val_norm,X_test_norm,scaler

def Normalize_full(X):
    from sklearn.preprocessing import StandardScaler
    X = X.astype(float)
    
    scaler = StandardScaler()
    
    X_norm = scaler.fit_transform(X)
    return X_norm,scaler

def Predict_full_model(clf,scaler,imputer,X,y):
        
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.dummy import DummyClassifier
    from sklearn import metrics
    
    X = scaler.transform(X)
    X = imputer.transform(X)
    
    # MODEL
    predictions = clf.predict_proba(X)[:,1]
    precision, recall, _ = precision_recall_curve(y, predictions)
    fpr, tpr, _ = metrics.roc_curve(y, predictions)
    auc = metrics.auc(fpr, tpr)
    ap = average_precision_score(y, predictions)
    print('AUC on validation set:',auc)
    print('Average precision on validation set:',ap)
   
    print('N feature vectors in validation fold:',len(y))
    print('Prevalence in validation fold:',sum(y)/len(y))
    
    return predictions,X,auc,ap

def prepare_feature_vectors(df,df_demo,ids_events,features,specs):
    print('prepare_feature_vectors triggered')
    
    """
    Samples feature vectors from the input dfs. 

    Parameters
    ----------
    df : pd.DataFrame
        df to sample from.
    df_demo: pd.DataFrame
        demograhics df to sample from.
    ids_events: list
        list with patient IDs which have the ICU trasnfer event
    features: np.array[str]
        Array of strings representing the names of the variables to be included in the model.
    specs: dict
        dictionary with model specifications
        
    Returns
    -------
    
    X: matrix [N feature vectors x N variables]
    y: vector [N feature vectors x 1]   type : np.array
    
    entry_dens_full [N feature vectors x N variables - n demographics]
    y_entry_dens [N feature vectors x 1]
    y_pat [N feature vectors x 1]
    y_t [N feature vectors x 1]
    
    """

    from datetime import datetime, timedelta
    import numpy as np    
    

    vectors = list() #create empty list for feature vectors
    
    # create empty lists for label (y), time-to-event (y_t) and patient label (y_pat)
    y_pat = list()
    y_t = list()
    y = list()
    ts = list()
    
    # empty lists for entry density
    entry_dens_full_pos = list()
    entry_dens_full_neg = list()
        
    # split df in pos and neg patients
    df_neg = df[~df['ID'].isin(ids_events)] # Isolate negative df --> no ICU 
    df_pos = df[df['ID'].isin(ids_events)]
    
    #ttransfer  dfs to numpy arrays, keep only ID, VARIABLE, TIME, VALUE, DEPARTMENT
    df_pos = df_pos[['ID','VARIABLE','TIME','VALUE','DEPARTMENT','DISCHARGE','START','ADMISSION']]
    df_neg = df_neg[['ID','VARIABLE','TIME','VALUE','DEPARTMENT','DISCHARGE','START','ADMISSION']]
    
    print('pos df:',df_pos.shape, '-->',len(df_pos['ID'].unique()), 'patients')
    print('neg df:',df_neg.shape, '-->',len(df_neg['ID'].unique()), 'patients')
    
    df_pos = np.asarray(df_pos)
    df_neg = np.asarray(df_neg)
    df_demo = np.asarray(df_demo)


    count = 0 # initiate counter for patients with LOS too short
    pos_counter = 0 # initiate counter of positive samples
    
    print('-----Sampling for positive patients-----') 
    
    n_pos_samples = int(specs['pred_window']/specs['int_pos'])
    print('max',n_pos_samples+1, 'positive samples per positive patient')
    
    for idx in np.unique(df_pos[:,0]):  # loop over pos patient IDs

        entry_dens_patient = list() # empty list for inividual entry density
    
        patient = df_pos[np.where(df_pos[:,0]==idx)] # isolate patient from df
        patient = patient[patient[:,2].argsort()]   # sort by date
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0] # isolate pat in dmographics df
        
        t_event = patient[np.where(patient[:,4]=='IC')][:,6].min() # define moment of ICU admission 
        
        a = patient[:,7]  # all admission dates
        a = pd.Series(a).dropna()
        s = patient[np.where(patient[:,4] != 'IC')][:,6] # all start dates outside ICU
        s = pd.Series(s).dropna()
        
        if (a.shape[0] ==0) and (s.shape[0] ==0):
            t_start = patient[:,2].min() # If nothing available, use firts measuremnet time
        
        elif (a.shape[0] ==0) and (s.shape[0] > 0):
            t_start = s.min()  # If no admission time avaliable but there is start time
        else:
            t_start = a.min() # If admission time is available use that
        
        total_los = np.round((t_event - t_start).total_seconds()/3600,0) # define total length-of-stay [hours]
        

        if total_los <= 0: # identify patients who's length of stay is negative
            count += 1
            print('los shorter than zero:',total_los,'pos episode:',idx)
            
        else:

            if total_los <= specs['int_pos']: # if LOS < the sampling interval:
                
                t = patient[:,2].min() + timedelta(hours=int(total_los/2))  #define time of smapling halfway the hospitalization
                los = np.round((total_los/2),1)
                temp = patient[np.where(patient[:,2]<t)] # filter temp for time of sampling
                
                if specs['entry_dens']:
                    ##### IF ENTRY DENISTY
                    t_2 = t - timedelta(hours=specs['int_pos'])
                    temp = temp[np.where(temp[:,2]>=t_2)]
                    ######
                else:
                    t_2 = t - timedelta(hours=specs['moving_feature_window']) # define earliest point to sample from
                    temp = temp[np.where(temp[:,2]>=t_2)] # filter temp for moving feature window
                
                if (los<=0) | (los> total_los):
                    print('wrong: negative (or zero) or too big los',los)
                    
                v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx) # Do actual sampling, collect v --> feature vector
                t_to_event = int((t_event - t).total_seconds() / 3600.0) 
                pos_counter += 1
                vectors.append(v) # collect all feature vectors in big list
                y_pat.append(1) # add patient label
                y_t.append(t_to_event)
                y.append(1)
                ts.append(t)
                entry_dens_patient.append(entry_dens) # collect all entry densities of single patient
                
                
            else: # if LOS > the sampling interval:
                los = specs['int_pos'] # start sampling after first interval 
                t = patient[:,2].min() + timedelta(hours=specs['int_pos']) # initialize t
                
                if specs['entry_dens_window']:
                    entry_dens_window = specs['entry_dens_window']
                else:
                    entry_dens_window = 10000
                
                runtime = np.min([total_los,entry_dens_window]) # define total time to sample from patient (total length of stay or specified)
                    
                while los < runtime:
                    
                    temp = patient[np.where(patient[:,2]<t)] # filter temp for time of sampling
                    
                    if specs['entry_dens']:
                        ##### IF ENTRY DENISTY
                        t_2 = t - timedelta(hours=specs['int_pos'])
                        temp = temp[np.where(temp[:,2]>=t_2)]
                        ######
                    else:
                        t_2 = t - timedelta(hours=specs['moving_feature_window'])
                        temp = temp[np.where(temp[:,2]>=t_2)]  # filter temp for moving feature window
                
                    if (los<0) | (los> total_los):
                        print('wrong: negative or too big los',los)
                
                    v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx) # Do actual sampling, collect v --> feature vector
                    t_to_event = int((t_event - t).total_seconds() / 3600.0)
                    
                    
                    if (t_to_event <= (specs['pred_window']+specs['gap'])) & (t_to_event > specs['gap']):
                    # IF time-to-event is smaller then prediction window, but larger than gap --> label positive
                    
                        pos_counter += 1
                        y.append(1)
                            
                    elif t_to_event > (specs['pred_window']+specs['gap']):
                    # IF time-to-event is larger then prediction window + gap --> label negative (Too early)
                    
                        y.append(0)
                    else:
                    # IF time-to-event is smaller then gap --> label negative (Too late)
                    
                        y.append(0)
                    
                    vectors.append(v) 
                    y_pat.append(1) # add patient label
                    y_t.append(t_to_event)
                    ts.append(t)
                    entry_dens_patient.append(entry_dens) # collect all entry densities of single patient
                        
                    los += specs['int_pos']
                    t = t + timedelta(hours=specs['int_pos'])
        
        # collect entry densities of all pos patients
        
        if np.array(entry_dens_patient).shape[0] > 0:
            entry_dens_patient = list(np.array(entry_dens_patient).mean(axis=0))
            entry_dens_full_pos.append(entry_dens_patient)
        else:
            print('No entry density for patient ',idx)
            
    print('number of pos patients with shorter stay than the defined GAP: ', count)    

          
    print('-----Sampling for negative patient-----')
    count=0
    for idx in np.unique(df_neg[:,0]): # loop over negative patients

        entry_dens_patient = list()
        
        patient = df_neg[np.where(df_neg[:,0]==idx)] # isolate df by patient ID
        patient = patient[patient[:,2].argsort()]   # sort by date
        demo = df_demo[np.where(df_demo[:,0]==idx)][:,1:][0]  # isolate demographics by patient ID
        
        if  pd.isnull(patient[:,2].min()):
            print('no times available')
        if pd.isnull(patient[:,5].min()):
            
            t_event = patient[:,2].max() # IF no discharge time available: event = last observed time
        else:            
            t_event = patient[:,5].min() # event = discharge time
        
        a = patient[:,7]  # all admission dates
        a = pd.Series(a).dropna()
        s = patient[np.where(patient[:,4] != 'IC')][:,6] # all start dates outside ICU
        s = pd.Series(s).dropna()
        
        if (a.shape[0] ==0) and (s.shape[0] ==0):
            t_start = patient[:,2].min() # If nothing available, use firts measuremnet time
        
        elif (a.shape[0] ==0) and (s.shape[0] > 0):
            t_start = s.min()  # If no admission time avaliable but there is start time
        else:
            t_start = a.min() # If admission time is available use that
        
        total_los = np.round((t_event - t_start).total_seconds()/3600,0) # define total length-of-stay [hours]
        
        if total_los <= 0: # identify patients who's length of stay is negative
            count += 1
            print('los nagetive or 0:',total_los,'neg episode:',idx)
            
        else:
            if total_los <= specs['int_neg']:  # if LOS < the sampling interval:
                
                t = patient[:,2].min() + timedelta(hours=int(total_los/2)) 
                los = np.round((total_los/2),1) # Update los with sampling interval
                temp = patient[np.where(patient[:,2]<t)]
                
                if specs['entry_dens']:
                    ##### IF ENTRY DENISTY
                    t_2 = t - timedelta(hours=specs['int_neg'])
                    temp = temp[np.where(temp[:,2]>=t_2)]
                    ######
                else:
                    t_2 = t - timedelta(hours=specs['moving_feature_window'])
                    temp = temp[np.where(temp[:,2]>=t_2)]
                    
                if (los<=0) | (los> total_los):
                    print('wrong: negative (or zero) or too big los',los)
                    
                v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx) # Do actual sampling, collect v --> feature vector
                t_to_event = int((t_event - t).total_seconds() / 3600.0)
                vectors.append(v)
                y_pat.append(0) # add patient label
                y_t.append(t_to_event)
                entry_dens_patient.append(entry_dens) # collect all entry densities of single patient
                y.append(0)
                ts.append(t)
                
            else:   # if LOS > the sampling interval:
                
                los =  specs['int_neg'] #initialize los
                t = patient[:,2].min() # initialize t
                
                if specs['entry_dens_window']:
                    entry_dens_window = specs['entry_dens_window']
                else:
                    entry_dens_window = 10000
                
                
                runtime = np.min([total_los,entry_dens_window])

                while los < runtime:
                
                    t = t + timedelta(hours=specs['int_neg'])
                    
                    temp = patient[np.where(patient[:,2]<t)]
                    if specs['entry_dens']:
                        ##### IF ENTRY DENISTY
                        t_2 = t - timedelta(hours=specs['int_neg'])
                        temp = temp[np.where(temp[:,2]>=t_2)]
                        ######
                    else:
                        t_2 = t - timedelta(hours=specs['moving_feature_window'])
                        temp = temp[np.where(temp[:,2]>=t_2)]
                
                    if (los<0) | (los> total_los):
                        print('wrong: negative or too big los',los)
                    
                    v,entry_dens = create_feature_window(temp,demo,features,los,specs,idx)
                    vectors.append(v) # add feature vector to 'neg' list
                    y_pat.append(0) # add patient label
                    y_t.append(int((t_event - t).total_seconds() / 3600.0))
                    entry_dens_patient.append(entry_dens)  # collect all entry densities of single patient
                    y.append(0)   
                    los += specs['int_neg']
                    ts.append(t)
         
        # collect entry densities of all neg patients
        if np.array(entry_dens_patient).shape[0] > 0:
            entry_dens_patient = list(np.array(entry_dens_patient).mean(axis=0))
            entry_dens_full_neg.append(entry_dens_patient)
        else:
            print('No entry density for patient ',idx)

        
    print('number of neg patients with shorter stay than the defined GAP: ', count)            
    
    print('N positive samples:',pos_counter)
    
    X = np.array([np.array(x) for x in vectors]) # create X by stacking all feature vectors
    print('X shape:',X.shape)
    
    y_pat = np.array([np.array(x) for x in y_pat])
    y_t = np.array([np.array(x) for x in y_t])
    y = np.array([np.array(x) for x in y])
    ts = np.array([np.array(x) for x in ts])
    
    print('y shape:',y.shape)
    assert(np.isnan(y).any() == False)
    assert(y.shape == y_pat.shape == y_t.shape == ts.shape)
    
    entry_dens_full_pos = np.array(entry_dens_full_pos)
    entry_dens_full_neg = np.array(entry_dens_full_neg)
    print('entry_density positive patients:',entry_dens_full_pos.shape)
    print('entry_density negative patients:',entry_dens_full_neg.shape)
    entry_dens_full = np.concatenate([entry_dens_full_pos,entry_dens_full_neg],axis = 0) # merge entry densities
    y_entry_dens = np.concatenate([np.ones(entry_dens_full_pos.shape[0]),np.zeros(entry_dens_full_neg.shape[0])],axis=0)
    print(entry_dens_full.shape)
    
    
    return X, y,entry_dens_full,y_pat,y_t,y_entry_dens,ts             
        

    
def create_feature_window(df,demo,variables,los,specs,idx):
    """
    Samples feature vectors from the input dfs. 

    Parameters
    ----------
    df : pd.DataFrame
        df with data of inidividual patient until moment of sampling
    demo: pd.DataFrame
        demograhics df to sample from.
    variables: np.array[str]
        Array of strings representing the names of the variables to be included in the model.
    los: int
        LOS of patient at moment of smapling
    specs: dict
        model specifications
    idx: str
        patient ID
        
    Returns
    -------
    v: feature vector, type: list
    entry_dens: entry denisty vector, type: list
    
    """
    
    import datetime as dt
    from datetime import datetime, timedelta
    
    
    variables = list(variables)
    variables.remove('FiO2') # remove FiO2 if present
    
    v = list() #define empty feature vector
    entry_dens = list()
    
    
    n = specs['feature_window'] # Feature window
    info_miss_variables = ['SpO2','HR','BP','RR','Temp'] #variables to sample info-missingness features from
    stat_features = ['SpO2','HR','BP','RR','po2_arterial','pco2_arterial','ph_arterial','pao2_over_fio2','base_excess'] #variables to sample stats features from
    # stat_features = ['SpO2','RR'] #variables to sample stats features from
    
    # ------ Add demographics  ---------
    
    for i in range(len(demo)): # -1 here causes AGE to be skipped
        if np.isnan(demo[i]):
            v.append(np.nan)
        else:
            v.append(demo[i])

    # -------- Add LOS ----------------
    if specs['time']:
        v.append(los)
    
    # ----- Add binary / continuous / ratio for NIV --------
    if specs['NIV']:
        temp = df[np.where(df[:,1]=='FiO2')]
        temp_sat = df[np.where(df[:,1]=='SpO2')]
        if temp.shape[0] < 1:
            v.append(0)
            entry_dens.append(1)
            v.append(np.nan)
            entry_dens.append(0)
        else:
            v.append(1)
            v.append(temp[-n:,3])
            entry_dens = entry_dens + list([1,1])
        if (temp.shape[0] > 0) & (temp_sat.shape[0]>0):
            v.append(temp_sat[-n,3]/temp[-n:,3])
            entry_dens.append(1)
        else:
            v.append(np.nan)
            entry_dens.append(0)
    
    # ------ Add Raw vairables (labs / vitals) ---------------
    count=0
    for item in variables: #loop over features
        
        temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
        
        if temp.shape[0] < 1: # If snippet contains none for this feature
            v.append(np.nan)
            entry_dens.append(0)
    
        else: #if snippet contains n or more values, take n most recent values
            v.extend(temp[-n:,3])          
            entry_dens.append(1)
            
        count+=1
   
    
    # -------- Add info_missingness ----------------

    if specs['freq']: # variable frequency
        
        for item in info_miss_variables: 
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
            v.append(temp.shape[0]/los) # number of instances / number of staying hours
            entry_dens.append(1)
            
    if specs['inter']: # variable interval
       
        for item in info_miss_variables: 
            
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
                
            # interal between current and previous measurement
            if temp.shape[0] > 1:
                v.append(np.round((temp[-1,2]-temp[-2,2]).total_seconds()/3600.0,1)) # Time between most recent and second most recent instance
                entry_dens.append(1)
            
            else: # less than 2 samples available?, no interval possible
                v.append(np.nan)
                entry_dens.append(0)
    
                
    # ---------- Add time of assessment ---------
    if specs['clocktime']:
        for item in info_miss_variables: 
            temp = df[np.where(df[:,1]==item)] # Extract snippet with only this feature
                
            # get clockhour of time of assessment
            if temp.shape[0] > 1:
                # print(temp[-1,2])
                # print(temp[-1,2].hour)
                v.append(temp[-1,2].hour)
                entry_dens.append(1)
            
            else: # less than 2 samples available?, no interval possible
                v.append(np.nan)
                entry_dens.append(0)
    
    # ---------- Add trend features ---------
    
    # Diff current - previous
    if specs['diff']:
        stat_features_present = [x for x in list(variables) if x in stat_features]
      
        for item in stat_features_present:
            
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
                # Add std of 1st order derivative
                v.append(np.std(np.diff(temp[:,3]))) # add diff_std
                entry_dens.append(1)
                # Add 2nd order derivative
                v.append(np.diff(np.diff(temp[-3:,3]))) # add diff_2
                entry_dens.append(1)
                
            else: # less than 3 samples available?, no std of diffs
                v.extend(np.ones(2)*np.nan)
                entry_dens = entry_dens + list([0,0])
            
            # if temp.shape[0] >= 4:
            #     # Add std of 2nd order derivative
            #     v.append(np.std(np.diff(np.diff(temp[:,3])))) # add diff_2_std
            #     entry_dens.append(1)
                
            # else: # less than 4 samples available?, no std of 2nd order diff
            #     v.extend(np.ones(1)*np.nan)
            #     entry_dens.append(0)
            
    v.append(idx) # finally add patient ID to feature vector (need this to do data split later)
    
    return v,entry_dens



def Imputer_full_model(X,specs):
    if specs['imputer'] == 'KNN':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=specs['knn'])
    
    elif specs['imputer'] == 'BR':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=0,initial_strategy=specs['initial_strategy'])
    
    elif specs['imputer'] == 'median':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy=specs['imputer'])    
    
    X_imp = imputer.fit_transform(X)
    return X_imp,imputer

def Imputer(X_train,X_val,X_test,specs):
    if specs['imputer'] == 'KNN':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=specs['knn'])
    
    elif specs['imputer'] == 'BR':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=0,initial_strategy=specs['initial_strategy'])
    
    elif specs['imputer'] == 'median':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy=specs['imputer']) 
        
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    print('NaNs imputed with', specs['imputer'])
    
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


    
def train_full_model(X,y,model,n_trees):
    
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
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import brier_score_loss
    
    if model == 'RF':
        print('start Random Forest')
        # define search space
        max_features = ['auto', 'log2','sqrt'] # Number of features to consider at every split
        max_depth = [2,3,5] # Maximum number of levels in tree, make not to deep to prevent overfitting
        # max_features = list(range(1,X.shape[1])) 
        param_grid = {  'max_features': max_features,
                         'max_depth': max_depth,
    
                        }
         
        clf = RandomForestClassifier(class_weight = 'balanced',verbose=0,n_estimators=n_trees)
       
    elif model == 'LR':
        print('start Logistic regression')
        param_grid = {'penalty':['l1', 'l2'],'C': np.logspace(-4, 4, 20),
                           'solver': ['liblinear']}
        clf = LogisticRegression(max_iter=100,class_weight = 'balanced',verbose=0)
        
    elif model == 'NB':
           print('modeling with Naive Bayes')
           from sklearn.naive_bayes import GaussianNB
           param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
           clf = GaussianNB()
            
    cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=1, random_state=random.randint(0, 10000))
    search = BayesSearchCV(estimator=clf,scoring='roc_auc',n_iter=50,search_spaces=param_grid, n_jobs=-1, cv=cv)   
        
    startTime = timer.time()
    search.fit(X, y)
    executionTime = (timer.time() - startTime)
    print('Execution time for hyper optimization:',str(executionTime))
    # report the best result
    print('best Hyperparams after optiomization:',search.best_params_)
        
    clf = search.best_estimator_
      
    
    print('Performance on Train set:')
    # Independend of threshold
    predictions = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, predictions)
    ap = average_precision_score(y, predictions)
    print('AUC: ',auc, ' AP:', ap)
    
    if model == 'LR' or model == 'NB':
        explainer = None
    else:
        explainer = shap.TreeExplainer(clf)
    
    return clf, explainer,auc
    
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
           max_depth = [2,3] # Maximum number of levels in tree, make not to deep to prevent overfitting
           
           param_grid = {  'max_features': max_features,
                           'max_depth': max_depth
                          }
           
           clf = RandomForestClassifier(class_weight = class_weight,verbose=0,n_estimators=n_trees)
          
       elif model == 'NB':
           print('modeling with Naive Bayes')
           from sklearn.naive_bayes import GaussianNB
           param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
           clf = GaussianNB()
       
       elif model == 'SVM':
           print('modeling with support vector machine')
           from sklearn import svm
           Cs = [0.001, 0.01, 0.1, 1, 10]
           gammas = [0.001, 0.01, 0.1, 1]
           param_grid = {'C': Cs, 'gamma' : gammas}
           clf = svm.SVC(kernel='rbf',probability=True)
           
       
       elif model == 'LR':
           
           param_grid = {'penalty':['l1', 'l2'],'C': np.logspace(-4, 4, 20),
                          'solver': ['liblinear']}
           clf = LogisticRegression(max_iter=100,class_weight = 'balanced',verbose=0)
       
           
       
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
       search = BayesSearchCV(estimator=clf,scoring='roc_auc',n_iter=25,search_spaces=param_grid, n_jobs=-1, cv=cv)   
       
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
       base_auc,base_ap,pred_tr,y_tr = evaluate_metrics(clf, X_test, y_test,X_train,y_train)
       
       clf_opt = search.best_estimator_
       print('Perfromance on test set with optimized model:')
       opt_auc,opt_ap,opt_pred_tr,opt_y_tr = evaluate_metrics(clf_opt, X_test,y_test,X_train,y_train)
       
       print('Improvement of {:0.2f}%.'.format( 100 * (opt_auc - base_auc) / opt_auc))
       
       
       # Pick optimized model
       clf_ret = clf_opt
       ap_ret = opt_ap
       auc_ret = opt_auc
       pred_tr_ret = opt_pred_tr
       y_tr_ret = opt_y_tr
           
       
       # #Pick the model with best performance on the test set
       # if opt_auc > base_auc:
       #     clf_ret = clf_opt
       #     ap_ret = opt_ap
       #     auc_ret = opt_auc
       #     pred_tr_ret = opt_pred_tr
       #     y_tr_ret = opt_y_tr
           
       # else:
       #     clf_ret = clf
       #     ap_ret = base_ap
       #     auc_ret = base_auc
       #     pred_tr_ret = pred_tr
       #     y_tr_ret = y_tr
           
   if (model == 'RF') or (model == 'XGB'):    
       explainer = shap.TreeExplainer(clf_ret)
   elif model == 'SVM':
       explainer = shap.KernelExplainer(clf_ret.predict,X_train)
   elif model == 'LR' or model == 'NB':
       explainer = None
     
   return clf_ret,explainer,auc_ret,ap_ret,pred_tr_ret,y_tr_ret,search.best_params_

def predict(model, test_features):
    print('predict triggered')
    
    predictions = model.predict_proba(test_features)[:,1]
    
    return predictions


    
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
        predictions_train = model.predict(train_features)
    else:
        predictions_train = model.predict_proba(train_features)[:,1]
        
    auc = roc_auc_score(train_labels, predictions_train)
    ap = average_precision_score(train_labels, predictions_train)
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
    
    
    return auc,ap,predictions_train,train_labels
    

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
            
def MEWS(sat,HR,BP,RR,temp):
    news = 0 #initalize MEWS score
    
    # resp rate
    if RR >= 30:
        news+=3
    elif RR <= 8 or (RR >= 21 and RR <= 29):
        news += 2
    elif RR == 9 or RR == 19 or RR == 20:
        news += 1
    
    # SpO2
    if sat <= 91:
        news+= 3
    elif sat == 92 or sat == 93:
        news+= 2
    elif sat == 94 or sat == 95:
        news+= 1
    
    #temp
    if temp <= 35 or temp >= 38.5:
        news+= 2
        
    # Bp
    if BP <= 70:
        news += 3
    elif (BP >= 71 and BP <= 80) or (BP >= 200):
        news += 2
    elif BP >= 81 and BP <= 100:
        news+= 1
    
    # HR
    if HR >= 130:
        news+= 3
    elif (HR <= 39) or (HR >= 111 and HR <= 129):
        news+= 2
    elif (HR >= 40 and HR <= 50) or (HR >= 101 and HR <= 110):
        news+=1
    
    # AVPU is assmued to be 'Alert' 
    
    return news 

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
        
    
def build_news(X_train,X_val,specs):
    
    # first merge val and test set
    # X_train =  np.concatenate((X_train, X_test), axis=0)
    
    # EWS(RR,sat,temp,BP,HR,oxy=False,AVPU =False):
    news_train = []
    df = X_train
    df = X_val
    news_val = []
    
    n = specs['feature_window']
    n_demo = len(specs['n_demo'])
    if specs['time']:
        n_demo += 1
    score = specs['NEWS']
    
    if score == 'NEWS':
        for i in range(df.shape[0]):
            news_train.append(NEWS(df[i,n_demo+(n-1)],      #SpO2
                                   df[i,1+n_demo+(n-1)],    #HR
                                   df[i,2+n_demo+(n-1)],    #BP
                                   df[i,3+n_demo+(n-1)],    #Resp
                                   df[i,4+n_demo+(n-1)]))   #Temp
        
        
            news_val.append(NEWS(df[i,n_demo+(n-1)],
                                   df[i,1+n_demo+(n-1)],
                                   df[i,2+n_demo+(n-1)],
                                   df[i,3+n_demo+(n-1)],
                                   df[i,4+n_demo+(n-1)]))
    
    elif score == 'MEWS':
        for i in range(df.shape[0]):
            news_train.append(MEWS(df[i,n_demo+(n-1)],      #SpO2
                                   df[i,1+n_demo+(n-1)],    #HR
                                   df[i,2+n_demo+(n-1)],    #BP
                                   df[i,3+n_demo+(n-1)],    #Resp
                                   df[i,4+n_demo+(n-1)]))   #Temp
        
        
            news_val.append(MEWS(df[i,n_demo+(n-1)],
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
    features = list(features)
    features.remove('FiO2')
    
    if (demo) & (specs['time']) & (specs['NIV']):
        total_features = [
                          # 'SEX',
                          'BMI',
                           # 'AGE',
                          'LOS']
    
    elif (demo) & (specs['time']):
        total_features = [
                          # 'SEX',
                          'BMI',
                           # 'AGE',
                          'LOS']
    elif demo:
        total_features = [
            # 'SEX',
            'BMI',
                           # 'AGE'
                          ]
    else:
        total_features = []
    
    if specs['NIV']:
        print('NIV added')
        total_features = total_features + ['NIV','FiO2','SpO2/FiO2']
    
    total_features = total_features + features
    
    if specs['freq']:
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:#features:
            new_features.append(i+str('_freq'))
        
        total_features.extend(new_features)
    if specs['inter']:
        new_features = []
        for i in ['SpO2','HR','BP','RR','Temp']:#features:
            new_features.append(i+str('_inter'))
        
        total_features.extend(new_features)
    
    if specs['clocktime']:
        new_features = []
        for i in features:
            new_features.append(i+str('_ass_time'))
        
        total_features.extend(new_features)
    
    if specs['diff']:
        new_features = []
        for i in ['SpO2','HR','BP','RR']:
            new_features.append(i+str('_signed_diff'))
        
        total_features.extend(new_features)
    if specs['stats']:
        stats = ['_max','_min','_mean','_median','_std','_diff_std','_signed_diff_2']#,'_diff_2_std']
        
        stat_features = ['SpO2','HR','BP','RR']
        # stat_features = ['SpO2','RR']
        
        for i in stat_features:
            new_features = []
            for stat in stats:
                new_features.append(i+stat)
            
            total_features.extend(new_features)
    total_features = np.asarray(total_features)
    print(total_features.shape)
    return total_features    

def Utility_score(y_pred,y_true,y_t,y_pat,t,specs):
    
    """
    Make Utilty score
    """
    
    mask = y_pred > t
    y_pred = mask.astype(int)
    print('n positive predictions:',sum(y_pred))
    
    U_P = np.concatenate([np.linspace(-0.1,1,72),
                           np.ones(specs['pred_window']),np.linspace(1,0.5,specs['gap'])],axis=0)
    U_N = np.concatenate([np.zeros(1000),np.linspace(0,-2,specs['pred_window']+specs['gap'])],axis=0)
    
    U_total = 0
    
    # nagetive patients:U_N.
    mask = y_pat == 0
    y_pred_neg = y_pred[mask].reset_index(drop=True)
    
    
    for i in range(y_pred_neg.shape[0]):

        U = 0 # TN
        
        if y_pred_neg[i] == 1: # FP
            U = -0.1


        U_total += U
    
    # positive patients
    mask = y_pat == 1
    y_pred_pos = y_pred[mask].reset_index(drop=True)
    y_true_pos = y_true[mask].reset_index(drop=True)
    y_t_pos = y_t[mask].reset_index(drop=True)

    
    for i in range(y_pred_pos.shape[0]):

        if (y_true_pos[i] == 0) & (y_pred_pos[i] == 0) & (y_t_pos[i] > (specs['gap']+specs['pred_window'] + 72)): # TN
            U = 0
        elif (y_true_pos[i] == 0) & (y_pred_pos[i] == 1) & (y_t_pos[i] > (specs['gap']+specs['pred_window'] + 72)): # FP
            U = -0,1
        
        else:
            if y_pred[i] == 1: # TP
                if y_t_pos[i] == 0:
                    U = U_P[-1]
                else:
                    U = U_P[int(-y_t_pos[i])]
                    
            elif y_pred[i] == 0: # FN
                if y_t_pos[i] == 0:
                    U = U_N[-1]
                else:
                    U = U_N[int(-y_t_pos[i])]
            
            U_total += U    
        
    return U_total

def plot_Utility(Y,specs):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    y_true  = Y.label
    y_pred = Y.pred
    y_t = Y.t_to_event
    y_pat = Y.patient
    
    y_no_pred = pd.Series(np.zeros(Y.shape[0]))
    print(y_true.shape)
    print(y_pred.shape)
    print(y_t.shape)
    print(y_no_pred.shape)
    print(y_pat.shape)
    
            
    mask = y_pat == 1
    U_opt = sum(mask)
    
    t_space = np.linspace(0,1,20)
    # t_space = np.arange(0,20,1)
    Us = []
    senses = []
    precs = []
    NPVs = []
    Joudens = []

    for t in t_space:
        print('threshold:',t)
        U_no_pred = Utility_score(y_no_pred,y_true,y_t,y_pat,t,specs) # Get Utility for model that never triggers
        print('U_no_pred:',U_no_pred)
            
        U_total = Utility_score(y_pred,y_true,y_t,y_pat,t,specs)
        print('U_total',U_total)
        U_norm = (U_total - U_no_pred)/(U_opt - U_no_pred)
        print('U_norm',U_norm)
        Us.append(U_norm)
        
        
        mask = y_pred > t
        y_pred_temp = mask.astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
        sens = tp / (tp+fn)
        senses.append(sens)
        prec = tp / (tp+fp)
        precs.append(prec)
        NPV = tn / (fn+tn)
        NPVs.append(NPV)
        spec = tn / (tn+fp)
        Jouden = sens+spec-1
        Joudens.append(Jouden)

        
        
    print(Us)
    plt.figure()
    plt.plot(t_space,Us,'r')
    plt.plot(t_space,senses,'b')
    plt.plot(t_space,precs,'c')
    plt.plot(t_space,NPVs,'g')
    
    # opt_t = t_space[np.where(Us == max(Us))]
    # plt.axvline(x = opt_t,color='r', linestyle='--')
    
    opt_jouden = t_space[np.where(Joudens == max(Joudens))]
    plt.axvline(x = opt_jouden,color='k', linestyle='--')
    
    # opt_prec = precs[np.where(t_space==opt_t)[0][0]]
    # plt.axhline(y=opt_prec,color='c', linestyle='--')
    # plt.text(0,opt_prec+0.03,str(np.round(opt_prec,3)),color='c',fontsize=9)
    
    # opt_sens = senses[np.where(t_space==opt_t)[0][0]]
    # plt.axhline(y=opt_sens,color='b', linestyle='--')
    # plt.text(0,opt_sens-0.06,str(np.round(opt_sens,3)),color='b',fontsize=9)
    
    jouden_prec = precs[np.where(t_space==opt_jouden)[0][0]]
    plt.axhline(y=jouden_prec,color='c', linestyle='--')
    plt.text(0,jouden_prec+0.03,str(np.round(jouden_prec,3)),color='c',fontsize=9)
    
    jouden_sens = senses[np.where(t_space==opt_jouden)[0][0]]
    plt.axhline(y=jouden_sens,color='b', linestyle='--')
    plt.text(0,jouden_sens+0.03,str(np.round(jouden_sens,3)),color='b',fontsize=9)
    
    plt.ylim(-0.4,1)
    plt.xlabel("thresholds")
    plt.legend(['Normalized Utility score','sensitivity','PPV','NPV'
                # ,'max Uility'
                ,"Max Youden's J"
                ], prop={'size': 7},loc='lower_right')
    
    # plt.ylabel("Normalized Utility score")
    plt.title('Utility Curve')
    plt.savefig('Utility_curve',dpi=300)
    
    # plt.figure()
    # # plt.plot(t_space,NPVs,'r')
    # plt.plot(t_space,precs,'b')
    # plt.savefig('NPV_PPV',dpi=300)

def mercaldo(sens,spec,prev):
    NPV = (spec*(1-prev))/((1-sens)*prev+spec*(1-prev))
    PPV = (sens*prev) / (sens*prev+(1-spec)*(1-prev))
    return NPV,PPV
   
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
    
    
    n = sum((df.LOS<=8))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>8) & (df.LOS<=16))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>16) & (df.LOS<=24))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>24) & (df.LOS<=72))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>72) & (df.LOS<=240))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
    n = sum((df.LOS>240))
    ex.append(str(n) + ' ('+ str(np.round(n/df.shape[0]*100,1))+ '%)')
        
    return ex