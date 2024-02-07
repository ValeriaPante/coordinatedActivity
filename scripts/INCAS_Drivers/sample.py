data_dir = "/scratch1/ashwinba/consolidated/INCAS/phase_2/TA2_eval_set_2024-01-27.jsonl.gz"
import pandas as pd
import os
import gzip
import json

df  = pd.read_json(data_dir,lines=True,compression='gzip')


print(df.shape)
print(df.columns)
print(df.head(5))