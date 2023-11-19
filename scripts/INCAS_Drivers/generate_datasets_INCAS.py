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
    file_root_dir = "/scratch1/ashwinba/consolidated/INCAS/"
    root_dir = "/scratch1/ashwinba/INCAS/sample_0908"
    

    finalDataFrame = pd.DataFrame()

    #sources = ['tumblr', 'facebook', 'reddit', 'twitter']
    sources  = ['twitter']

    for fileDir in fileDirs:
        source_lst = list(filter(lambda x: x in fileDir, sources))
        if(len(source_lst)>=1):
            source = source_lst[0]
            df = pd.read_json(path_or_buf=os.path.join(root_dir,fileDir), lines=True)
            #print(df.columns)
            #print(df.isnull().sum())
            #print(df.head(2)['mediaTypeAttributes'].values)
            # False Id
            if(source == 'twitter'):
                df['author'] = df['mediaTypeAttributes'].apply(lambda x:dict(x)['twitterData']['twitterAuthorScreenname'])
                df['tweetid'] = df['mediaTypeAttributes'].apply(lambda x:dict(x)['twitterData']['tweetId'])
                df['retweet_id'] = df['mediaTypeAttributes'].apply(lambda x:dict(x)['twitterData']['engagementParentId'])
                df['engagementType'] = df['mediaTypeAttributes'].apply(lambda x:dict(x)['twitterData']['engagementType'])
                
            
            lists = df['mediaTypeAttributes'].apply(lambda x:list(dict(x)['twitterData'].keys()))
            lists_sum = sum(lists,[])
            print(set(lists_sum))
            
            print(df['engagementType'].value_counts())

            # Dropping empty user ids          
            df.dropna(subset=['author'],inplace=True)

            # Removing unecessary columns
            df.drop(['translatedTitle','translatedContentText','geolocation'],inplace=True,axis=1)
            
            df['source_data'] = source
            
            #finalDataFrame = finalDataFrame._append(df,ignore_index=True)
            finalDataFrame = pd.concat([finalDataFrame,df], ignore_index=True)
            print(finalDataFrame.shape)
    
            warnings.warn("Completed for {SOURCE}".format(SOURCE=source))
        
    
    author_mappings = dict(zip(list(finalDataFrame.author.unique()), list(range(finalDataFrame.author.unique().shape[0])))) 

    # Userid
    finalDataFrame['userid'] = finalDataFrame['author'].apply(lambda x: author_mappings[x]).astype(int)
    warnings.warn("after label encoding")
    
    # Sorting cum based on timePublished
    finalDataFrame.sort_values(by=['timePublished'], inplace=True)

    print(finalDataFrame.columns)

    #print(df['geolocation'].unique)

    warnings.warn("File Consolidated")
    finalDataFrame.to_csv(os.path.join(
        file_root_dir, "consolidated_INCAS_0908.csv.gz"),index=False)
    warnings.warn("Consolidated File Saved")


root_dir = "/scratch1/ashwinba/INCAS/sample_0908"
files_dirs = os.listdir(root_dir)
print(files_dirs)

GenerateDatasets(files_dirs)
