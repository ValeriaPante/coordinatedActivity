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
        print(finalDataFrame.isna().sum())

        warnings.warn("Completed for {SOURCE}".format(SOURCE=source))

    warnings.warn("File Consolidated")
    finalDataFrame.to_csv(os.path.join(
        file_root_dir, "consolidated_INCAS.csv"),index=False)
    warnings.warn("Consolidated File Saved")


root_dir = "/scratch1/ashwinba/INCAS"
files_dirs = os.listdir(
    root_dir)

GenerateDatasets(files_dirs)

file = pd.read_csv(os.path.join("/scratch1/ashwinba/consolidated/INCAS/","consolidated_INCAS.csv"))
file.dropna(subset=['title'],inplace=True)
print(file.shape)
print(file.head(20))
print(file.isna().sum())
