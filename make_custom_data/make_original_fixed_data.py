import numpy as np
import pandas as pd
import os
import random

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

DATA_PATH = '/opt/ml/input/data/train_dataset'
train_org_df = pd.read_csv(os.path.join(DATA_PATH, "train_data_add_elapsed.csv"), dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

u_id = train_org_df.userID.unique()
random.seed(0)
random.shuffle(u_id)
size = int(len(u_id) * 0.2)
u_id = u_id[:size]

train_org_df.loc[~train_org_df.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "original_fixed_train.csv"), index=False)
train_org_df.loc[train_org_df.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "original_fixed_valid.csv"), index=False)

print(f"make original_fixed_data done")