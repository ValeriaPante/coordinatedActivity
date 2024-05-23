import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import warnings

le = LabelEncoder()

def find_author(user_dict):
    if('twitterAuthorScreenname' in list(user_dict.keys())):
        return user_dict['twitterAuthorScreenname']
    return np.nan

def find_tweetId(user_dict):
    if('tweetId' in list(user_dict.keys())):
        return user_dict['tweetId']
    print(user_dict)
    return np.nan

def find_author_name(title):
    name = re.findall(r"\w*[!@#$%^&*]+\w*",title)
    if(len(name) >=1):
        return name[0][1:]
    return np.nan

def find_retweet(user_dict):
    if(user_dict['engagementType'] == 'retweet'):
        return user_dict['engagementParentId']
    return np.nan

def GenerateDatasets(fileDirs):
    # Root Directory
    file_root_dir = "/scratch1/ashwinba/consolidated/INCAS/phase_2"
    finalDataFrame = pd.DataFrame()

    sources  = ['twitter']

    df = pd.read_json(fileDirs[0], lines =True,compression='gzip')
 
    # label_encoding columns
    le_dict = {'mediaTypeAttributes.twitterData.twitterAuthorScreenname':'author','mediaTypeAttributes.twitterData.tweetId':'tweetid','mediaTypeAttributes.twitterData.engagementParentId':'retweet_id'}

    for key,value in le_dict.items():
        df[value] = le.fit_transform(df[key].values)

   # Renaming Columns
    rename_dict={'mediaTypeAttributes.twitterData.engagementType':'engagementType'}
    
    for key,value in rename_dict.items():
        df[value] = df[key].values

    #df.drop(list(rename_dict.keys()),axis=1,inplace=True)

    #finalDataFrame = finalDataFrame._append(df,ignore_index=True)
    finalDataFrame = df.copy()
    del df

    author_mappings = dict(zip(list(finalDataFrame['author'].unique()), list(range(finalDataFrame['author'].unique().shape[0])))) 

    # Userid
    finalDataFrame['userid'] = finalDataFrame['author'].apply(lambda x: author_mappings[x]).astype(int)
    warnings.warn("after label encoding")
    
    # Sorting cum based on timePublished
    finalDataFrame.sort_values(by=['timePublished'], inplace=True)

    warnings.warn("File Consolidated")
    finalDataFrame.to_csv(os.path.join(
        file_root_dir, "processed_INCAS_TA2.csv.gz"),index=False)
    warnings.warn("Consolidated File Saved")

root_dir = "/scratch1/ashwinba/INCAS/sample_0908"
#files_dirs = os.listdir(root_dir)
files_dirs = ['/scratch1/ashwinba/consolidated/INCAS/phase_2/TA2_sample_set_2024-01-18.jsonl.gz']

GenerateDatasets(files_dirs)
