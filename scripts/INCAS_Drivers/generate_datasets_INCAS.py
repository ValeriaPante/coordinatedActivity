import pandas as pd
import os
import numpy as np

import warnings


def GenerateDatasets(fileDirs):
    # Root Directory
    file_root_dir = "/scratch1/ashwinba/consolidated/INCAS/"
    root_dir = "/scratch1/ashwinba/INCAS"

    finalDataFrame = pd.DataFrame()

    sources = ['tumblr', 'facebook', 'reddit', 'twitter']

    for fileDir in fileDirs:
        source = list(filter(lambda x: x in fileDir, sources))[0]
        df = pd.read_json(path_or_buf=os.path.join(root_dir,fileDir), lines=True)
        
        # False Id
        if(source == 'twitter'):df['author'] = np.arange(1,df.shape[0]+1)
        
        # Removing unecessary columns
        df.drop(['translatedTitle','translatedContentText','geolocation','mentionedUsers'],inplace=True,axis=1)
        
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

    # FFill for NaN values
    #df['retweet_id'] = np.nan

    contextsCounts = finalDataFrame['contentText'].value_counts()
    repeatedContexts = contextsCounts.where(contextsCounts > 1)
    repeatedContexts.dropna(inplace=True)


    repeatedContexts = list(repeatedContexts.index)
    mask = finalDataFrame['contentText'].isin(repeatedContexts)
    df_mask = finalDataFrame[mask][['id','contentText']]
    df_mask.rename(columns={'id':'retweet_id'},inplace=True)
    df_mask.drop_duplicates(subset=['contentText'],inplace=True,keep='first')

    df = pd.merge(finalDataFrame,df_mask,on='contentText',how='outer')

    df.loc[(df['id'] == df['retweet_id']), "retweet_id"] = np.nan    
    
    print(df.shape)
    
    # Display no of repeatedCounts
    warnings.warn("repeatedCounts: "+str(len(repeatedContexts)))

    warnings.warn("File Consolidated")
    df.to_csv(os.path.join(
        file_root_dir, "consolidated_INCAS.csv.gz"),index=False)
    warnings.warn("Consolidated File Saved")


root_dir = "/scratch1/ashwinba/INCAS"
files_dirs = os.listdir(
    root_dir)

GenerateDatasets(files_dirs)
