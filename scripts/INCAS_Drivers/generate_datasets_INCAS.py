import pandas as pd
import os

import warnings


def GenerateDatasets(fileDirs):
    # Root Directory
    file_root_dir = "/scratch1/ashwinba/consolidated/INCAS/"

    finalDataFrame = pd.DataFrame()

    sources = ['tumblr', 'facebook', 'reddit', 'twitter']

    for fileDir in fileDirs:
        source = list(filter(lambda x: x in fileDir, sources))[0]
        df = pd.read_json(path_or_buf=fileDir, lines=True)
        df['source_data'] = source
        finalDataFrame = pd.concat([df, finalDataFrame], ignore_index=True)
        warnings.warn("Completed for {SOURCE}".format(SOURCE=source))

    warnings.warn("File Consolidated")
    finalDataFrame.to_csv(os.path.join(
        file_root_dir, "consolidated_INCAS.csv.gz"))
    warnings.warn("Consolidated File Saved")


root_dir = "/scratch1/ashwinba/INCAS"
files_dirs = os.listdir(
    root_dir)

GenerateDatasets(files_dirs)
