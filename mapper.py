import pandas as pd
import numpy as np
import os

import gzip

def get_negative_data(control_data,country_name):
    control_data['userid'] = control_data['user'].apply(lambda x: int(eval(x)['id']))
    if(country_name == 'uae'  or country_name == 'egypt'):
        control_data['country'] = "UAE_EGYPT"
    else:
        control_data['country'] = country_name

    control_data = control_data[['userid','country']]
    return control_data

def get_positive_data(treated_data,country_name):
    if(country_name == 'uae'  or country_name == 'egypt'):
        treated_data['country'] = "UAE_EGYPT"
    else:
        treated_data['country'] = country_name

    treated_data = treated_data[['userid','country']]
    return treated_data 

def GenerateDatasets(datasetsPaths): 
    for record in datasetsPaths:
        country_name,file = record
        if 'control' in file:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True),country_name)])
                except:
                    with gzip.open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True),country_name)
            else:
                try:
                    with open(file) as f:
                        control = pd.concat([control, get_negative_data(pd.read_json(f, lines =True),country_name)])
                except:
                    with open(file) as f:
                        control = get_negative_data(pd.read_json(f, lines =True),country_name)
        else:
            if file[-2:] == 'gz':
                try:
                    with gzip.open(file) as f:
                        treated = pd.concat([treated, get_positive_data(pd.read_csv(f),country_name)])
                except:
                    with gzip.open(file) as f:
                        treated = get_positive_data(pd.read_csv(f),country_name)
            else:
                try:
                    with open(file) as f:
                        treated = pd.concat([treated, get_positive_data(pd.read_csv(f),country_name)])
                except:
                    with open(file) as f:
                        treated = get_positive_data(pd.read_csv(f),country_name)
            
    pos_en_df_all = treated
    del treated
    neg_en_df_all = control
    del control

    pos_en_df_all.to_csv("/scratch1/ashwinba/consolidated/treated_consolidated_ids.csv.gz", index=False, compression='gzip')
    neg_en_df_all.to_csv("/scratch1/ashwinba/consolidated/control_consolidated_ids.csv.gz", index=False, compression='gzip')

root_dir = "/project/ll_774_951/InfoOpsNationwiseDriverControl"
countries_dir = os.listdir("/project/ll_774_951/InfoOpsNationwiseDriverControl")

dataset_dirs = []

for country in countries_dir:
    files_dir = os.listdir(os.path.join(root_dir,country))

    ## Country File Names Check
    control_check = list(filter(lambda x:"control" in x,files_dir))
    treated_check = list(filter(lambda x:"tweets_csv_unhashed" in x,files_dir))
    
    if(len(control_check) >= 1 and len(treated_check) >= 1):
        dataset_dirs.append([country,os.path.join(root_dir,country,treated_check[0])])
        dataset_dirs.append([country,os.path.join(root_dir,country,control_check[0])])

GenerateDatasets(dataset_dirs)



