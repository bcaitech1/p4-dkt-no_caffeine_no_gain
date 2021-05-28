import numpy as np
import pandas as pd
from tqdm import tqdm

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

DATA_PATH = '/opt/ml/input/data/train_dataset/train_data.csv'
train_org_df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

train_df = train_org_df.copy()

diff = train_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=-1))
diff = diff.fillna(pd.Timedelta(seconds=-1))
diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

train_df['elapsed'] = diff

tmp = ""
idx = []

for i in tqdm(train_df.index):
    if tmp == train_df.loc[i, "testId"]:
        continue
    else:
        tmp = train_df.loc[i, "testId"]
        idx.append(i)

train_df.loc[idx, "elapsed"] = 0
train_df.loc[train_df.elapsed > 250, "elapsed"] = 0
train_df.loc[train_df.elapsed == 0, "elapsed"] = train_df.loc[train_df.elapsed != 0, "elapsed"].mean()
train_df.to_csv("/opt/ml/input/data/train_dataset/train_data_add_elapsed.csv", index=False)

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

# 데이터 경로 맞춰주세요!
DATA_PATH = '/opt/ml/input/data/train_dataset/test_data.csv'
test_org_df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=['Timestamp'])
test_org_df = test_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

test_df = test_org_df.copy()

diff = test_df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=-1))
diff = diff.fillna(pd.Timedelta(seconds=-1))
diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

test_df['elapsed'] = diff

tmp = ""
idx = []

for i in tqdm(test_df.index):
    if tmp == test_df.loc[i, "testId"]:
        continue
    else:
        tmp = test_df.loc[i, "testId"]
        idx.append(i)

test_df.loc[idx, "elapsed"] = 0
test_df.loc[test_df.elapsed > 250, "elapsed"] = 0
test_df.loc[test_df.elapsed == 0, "elapsed"] = test_df.loc[test_df.elapsed != 0, "elapsed"].mean()
test_df.to_csv("/opt/ml/input/data/train_dataset/test_data_add_elapsed.csv", index=False)