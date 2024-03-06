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

def filter_media(record):
    if('twitterData' in record['mediaTypeAttributes'].keys()):
        return True
    return False
    
def update_vals(record):
    columns = ['twitterAuthorScreenname','tweetId','engagementParentId','engagementType']
    
    filtered_columns = list(filter(lambda x:x not in record['twitterData'].keys(),columns))

    for column in filtered_columns:
        record['twitterData'][column]  = np.NaN
        
    return record

    
def GenerateDatasets(fileDirs):
    # Root Directory
    file_root_dir = "/scratch1/ashwinba/consolidated/INCAS/phase_2"

    finalDataFrame = pd.DataFrame()

    for fileDir in fileDirs:
        warnings.warn("came inside loop")
        # df = pd.read_json(path_or_buf=os.path.join(root_dir,fileDir), lines=True)
        df = pd.read_json(path_or_buf=fileDir, lines=True,compression='gzip')
        warnings.warn("done with reading the dataframe")
        print(df.columns)
        df.dropna(subset=['mediaTypeAttributes'],inplace=True)
        print(df['mediaTypeAttributes'].values)
        
        
        # Filtering records where key exists
        df['mediaTypeAttributes'] = df['mediaTypeAttributes'].apply(lambda x:dict(x))
        df = df[df.apply(filter_media,axis=1)]
        df['mediaTypeAttributes'] = df['mediaTypeAttributes'].apply(lambda x:update_vals(x))
        
        # Extracting the features
        df['author'] = df['mediaTypeAttributes'].apply(lambda x:x['twitterData']['twitterAuthorScreenname'])
        df['tweetid'] = df['mediaTypeAttributes'].apply(lambda x:x['twitterData']['tweetId'])
        df['retweet_id'] = df['mediaTypeAttributes'].apply(lambda x:x['twitterData']['engagementParentId'])
        df['engagementType'] = df['mediaTypeAttributes'].apply(lambda x:x['twitterData']['engagementType'])
            
        warnings.warn("done with stage-1")
        
        print(df['engagementType'].value_counts())
    
        # Dropping empty user ids          
        df.dropna(subset=['author'],inplace=True)
    
        # Removing unecessary columns
        df.drop(['translatedTitle','translatedContentText','geolocation'],inplace=True,axis=1)

        #finalDataFrame = finalDataFrame._append(df,ignore_index=True)
        #finalDataFrame = pd.concat([finalDataFrame,df], ignore_index=True)
        finalDataFrame = df.copy()
        print(finalDataFrame.shape)
        
        del df

    warnings.warn("came out of loop")
    
    author_mappings = dict(zip(list(finalDataFrame.author.unique()), list(range(finalDataFrame.author.unique().shape[0])))) 

    # Userid
    finalDataFrame['userid'] = finalDataFrame['author'].apply(lambda x: author_mappings[x]).astype(int)
    warnings.warn("after label encoding")
    
    # Sorting cum based on timePublished
    finalDataFrame.sort_values(by=['timePublished'], inplace=True)

    print(finalDataFrame.columns)
    warnings.warn("File Consolidated")
    finalDataFrame.to_csv(os.path.join(
        file_root_dir, "consolidated_INCAS_EVAL_2.csv.gz"),index=False)

# root_dir = "/scratch1/ashwinba/INCAS/sample_0908"
# files_dirs = os.listdir(root_dir)
# print("files_dirs)

GenerateDatasets(["/scratch1/ashwinba/consolidated/INCAS/phase_2/TA2_small_eval_set_2024-02-16.jsonl.gz"])
