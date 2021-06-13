import numpy as np
import pandas as pd
import os
import math

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}   

# 데이터 경로 맞춰주세요!
DATA_PATH = '/opt/ml/input/data/train_dataset'
train_org_df = pd.read_csv(os.path.join(DATA_PATH, "train_data_add_elapsed.csv"), dtype=dtype, parse_dates=['Timestamp'])
train_org_df = train_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
test_org_df = pd.read_csv(os.path.join(DATA_PATH, "test_data_add_elapsed.csv"), dtype=dtype, parse_dates=['Timestamp'])
test_org_df = test_org_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

def feature_engineering(df):

    # 문항이 중간에 비어있는 경우를 파악 (1,2,3,,5)
    def assessmentItemID2item(x):
        return int(x[-3:]) - 1  # 0 부터 시작하도록 
    df['item'] = df.assessmentItemID.map(assessmentItemID2item)

    item_size = df[['assessmentItemID', 'testId']].drop_duplicates().groupby('testId').size()
    testId2maxlen = item_size.to_dict() # 중복해서 풀이할 놈들을 제거하기 위해

    item_max = df.groupby('testId').item.max()
    shit_index = item_max[item_max +1 != item_size].index
    shit_df = df.loc[df.testId.isin(shit_index),['assessmentItemID', 'testId']].drop_duplicates().sort_values('assessmentItemID')      
    shit_df_group = shit_df.groupby('testId')

    shitItemID2item = {}
    for key in shit_df_group.groups:
        for i, (k,_) in enumerate(shit_df_group.get_group(key).values):
            shitItemID2item[k] = i
        
    def assessmentItemID2item_order(x):
        if x in shitItemID2item:
            return int(shitItemID2item[x])
        return int(x[-3:]) - 1  # 0 부터 시작하도록 
    df['item_order'] =  df.assessmentItemID.map(assessmentItemID2item_order)



    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    # 유저가 푼 시험지에 대해, 유저의 전체 정답/풀이횟수/정답률 계산 (3번 풀었으면 3배)
    df_group = df.groupby(['userID','testId'])['answerCode']
    df['user_total_correct_cnt'] = df_group.transform(lambda x: x.cumsum().shift(1))
    df['user_total_ans_cnt'] = df_group.cumcount()
    df['user_total_acc'] = df['user_total_correct_cnt'] / df['user_total_ans_cnt']

    # 유저가 푼 시험지에 대해, 유저의 풀이 순서 계산 (시험지를 반복해서 풀었어도, 누적되지 않음)
    # 특정 시험지를 얼마나 반복하여 풀었는지 계산 ( 2번 풀었다면, retest == 1)
    df['test_size'] = df.testId.map(testId2maxlen)
    df['retest'] = df['user_total_ans_cnt'] // df['test_size']
    df['user_test_ans_cnt'] = df['user_total_ans_cnt'] % df['test_size']

    # 각 시험지 당 유저의 정확도를 계산
    df['user_test_correct_cnt'] = df.groupby(['userID','testId','retest'])['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_acc'] = df['user_test_correct_cnt']/df['user_test_ans_cnt']

    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_a = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
    correct_a.columns = ["ItemID_mean", 'ItemID_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']
    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_a, on=['assessmentItemID'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")

    return df

# 맞춰야하는 문항 ID 파악
set_assessmentItemID = set(test_org_df.loc[test_org_df.answerCode == -1, 'assessmentItemID'].values)

train = feature_engineering(train_org_df)
test = feature_engineering(test_org_df)
# 피처를 대충만들어서 꽤 오래걸립니다.

train = train.fillna(0)
test = test.fillna(0)

def caffeine_feature(df):
    
    df['testId'] = df['testId'].str[1:]
    df['assessmentItemID'] = df['assessmentItemID'].str[1:]
    # use 3 features instead testId, assessmentItemID
    df['classification'] = df['testId'].str[2:3]
    df['paperNum'] = df['testId'].str[-3:]
    df['problemNum'] = df['assessmentItemID'].str[-3:]

    df = df.astype({'Timestamp': 'datetime64[ns]', 'classification' : 'int', 'paperNum' : 'int', 'problemNum': 'int', 'assessmentItemID' : 'int'})

    def hours(timestamp):
        return int(str(timestamp).split()[1].split(":")[0])
    
    df["hours"] = df.Timestamp.apply(hours)
    
    def time_bin(hours):
        if 0 <= hours <= 5:
            # Night
            return 0
        elif 6 <= hours <= 11:
            # Morning
            return 1
        elif 12 <= hours <= 17:
            # Daytime
            return 2
        else:
            # Evening
            return 3
    
    df["time_bin"] = df.hours.apply(time_bin)
    df = df.astype({'Timestamp': 'str'})

    tag_df = df.copy()
    tag_df.sort_values(by=['KnowledgeTag', 'Timestamp'], inplace=True)

    # test의 정답률(모든 사용자 탐색)
    tag_df['knowledge_correct_answer'] = tag_df.groupby('KnowledgeTag')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    tag_df['knowledge_total_answer'] = tag_df.groupby('KnowledgeTag')['answerCode'].cumcount()
    tag_df['knowledge_acc'] = tag_df['knowledge_correct_answer']/tag_df['knowledge_total_answer']

    test = tag_df[tag_df['KnowledgeTag'] != tag_df['KnowledgeTag'].shift(-1)]
    tag_acc = {}
    for row in test.iterrows():
        tag_acc[row[1][5]] = row[1]["knowledge_acc"]  # -는 컬럼의 인덱스에 따라 변경
        
    df['tag_acc']  = tag_df['KnowledgeTag'].map(lambda x : tag_acc[x])
    
    assess_df = df.copy()
    assess_df.sort_values(by=['assessmentItemID', 'Timestamp'], inplace=True)

    assess_df['assessment_correct_answer'] = assess_df.groupby('assessmentItemID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    assess_df['assessment_total_answer'] = assess_df.groupby('assessmentItemID')['answerCode'].cumcount()
    assess_df['assessment_acc'] = assess_df['assessment_correct_answer']/assess_df['assessment_total_answer']

    test = assess_df[assess_df['assessmentItemID'] != assess_df['assessmentItemID'].shift(-1)]
    assessment_acc = {}
    for row in test.iterrows():
        assessment_acc[row[1][1]] = row[1]["assessment_acc"] # -에 assess_df가 있는 컬럼 인덱스
        
    df['assessment_acc']  = assess_df['assessmentItemID'].map(lambda x : assessment_acc[x])

    test_df = df.copy()
    test_df.sort_values(by=['testId', 'Timestamp'], inplace=True)

    # test의 정답률(모든 사용자 탐색)
    test_df['test_correct_answer'] = test_df.groupby('testId')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    test_df['test_total_answer'] = test_df.groupby('testId')['answerCode'].cumcount()
    test_df['test_acc'] = test_df['test_correct_answer']/test_df['test_total_answer']

    test = test_df[test_df['testId'] != test_df['testId'].shift(-1)]
    test_acc = {}
    for row in test.iterrows():
        test_acc[row[1][2]] = row[1]["test_acc"] # -에 test_df가 있는 컬럼 인덱스
        
    df['test_acc']  = test_df['testId'].map(lambda x : test_acc[x])

    df['time'] = pd.to_datetime(df['Timestamp']).astype(int)/ 10**17

    # 미래 정보
    df['correct_shift_-2'] = df.groupby('userID')['answerCode'].shift(-2)
    df['correct_shift_-1'] = df.groupby('userID')['answerCode'].shift(-1)

    # 과거 정보
    df['correct_shift_1'] = df.groupby('userID')['answerCode'].shift(1)
    df['correct_shift_2'] = df.groupby('userID')['answerCode'].shift(2)

    df['total_used_time'] = df.groupby('userID')['time'].cumsum()

    df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
    df['past_correct'] = df.groupby('userID')['shift'].cumsum()

    reversed_edu_correct_df = df.iloc[::-1].copy()

    # 미래에 맞출 문제 수
    reversed_edu_correct_df['shift'] = reversed_edu_correct_df.groupby('userID')['answerCode'].shift().fillna(0)
    reversed_edu_correct_df['future_correct'] = reversed_edu_correct_df.groupby('userID')['shift'].cumsum()
    df = reversed_edu_correct_df.iloc[::-1]


    df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
    df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()

    df['past_count'] = df.groupby('userID').cumcount()

    # 과거에 푼 문제 수
    df['past_count'] = df.groupby('userID').cumcount()

    # 과거에 맞춘 문제 수
    df['shift'] = df.groupby('userID')['answerCode'].shift().fillna(0)
    df['past_correct'] = df.groupby('userID')['shift'].cumsum()

    # 과거 평균 정답률
    df['average_correct'] = (df['past_correct'] / df['past_count']).fillna(0)


    # 과거에 해당 문제를 푼 수
    df['past_content_count'] = df.groupby(['userID', 'assessmentItemID']).cumcount()

    # 과거에 해당 문제를 맞춘 수
    df['shift'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
    df['past_content_correct'] = df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()

    # 과거 해당 문제 평균 정답률
    df['average_content_correct'] = (df['past_content_correct'] / df['past_content_count']).fillna(0)

    df['mean_time'] = df.groupby(['userID'])['time'].rolling(3).mean().values

    # 평균 (mean) / 표준 편차 (std)
    agg_df = df.groupby('userID').agg(['mean', 'std'])

    # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
    agg_dict = agg_df.to_dict()

    # 구한 통계량을 각 사용자에게 mapping
    for k, v in agg_dict.items():
        # feature 이름
        feature_name = "_".join(k)
        
        # mapping이후 새로운 feature 추가
        df[feature_name] = df['userID'].map(v)

    # 중간값 (median)
    agg_df = df.groupby('userID')['time'].agg(['median'])

    # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
    agg_dict = agg_df.to_dict()

    # 구한 통계량을 각 사용자에게 mapping
    df['time_median'] = df['userID'].map(agg_dict['median'])
    df['hour'] = df['time'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)
    
    hour_dict = df.groupby(['hour'])['answerCode'].mean().to_dict()
    df['correct_per_hour'] = df['hour'].map(hour_dict)

    df['hour'] = df['time'].transform(lambda x: pd.to_datetime(x, unit='s').dt.hour)

    # 사용자의 주 활동 시간
    mode_dict = df.groupby(['userID'])['hour'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
    df['hour_mode'] = df['userID'].map(mode_dict)

    # 사용자의 야행성 여부
    # 시간이 10 ~ 15시 사이에 분포되어 있어 여기에서는 임의로 12로 분리하였다
    df['is_night'] = df['hour_mode'] > 12

    df['normalized_time'] = df.groupby('userID')['time'].transform(lambda x: (x - x.mean())/x.std())

    df['relative_time'] = df.groupby('userID').apply(lambda x: x['time'] - x['time'].median()).values

    df['time_cut'] = pd.cut(df['time'], bins=3)
    df['time_qcut'] = pd.qcut(df['time'], q=3)

    def same_tag_fillna(s):
        try:
            int(s)
            return True
        except:
            return False

    df.loc[(df.userID==df.userID.shift(-1))&(df.KnowledgeTag==df.KnowledgeTag.shift(1)), "same_tag"] = True
    df["same_tag"] = df["same_tag"].apply(same_tag_fillna)

    cont = []
    cnt = 0
    tmp_uid = df.userID[0]
    for uid, tf in zip(df.userID, df.same_tag):
        if tf and uid==tmp_uid:
            cnt += 1
            cont.append(cnt)
        else:
            cnt = 0
            cont.append(cnt)
        if uid!=tmp_uid:
            tmp_uid = uid

    df["cont_tag"] = cont

    i = 0
    ll = []
    for x in df.columns:
        i += 1
        _sum = sum(df[x].isnull())
        # print(f"{x}: {_sum}")
        if _sum != 0:
            ll.append(x)
    
    def fillna(x):
        if math.isnan(x):
            return 0
        return x
    
    for x in ll:
        df[x] = df[x].apply(fillna)

    return df

train = caffeine_feature(train)
test = caffeine_feature(test)

acc = train.loc[train.userID!=train.userID.shift(-1), ["userID", "user_total_acc", "user_total_ans_cnt"]].sort_values("user_total_acc")

u_id = []
for i in range(len(acc.userID.values)):
    if i % 10 in [0, 5]:
        u_id.append(acc.userID.values[i])

print(f"valid로 들어가는 user 비율: {len(u_id) / acc.shape[0]}")
print(f"valid로 들어가는 row 비율: {train.loc[train.userID.isin(u_id), :].shape[0] / train.shape[0]}")

train.loc[~train.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "add_FE_fixed_train.csv"), index=False)
train.loc[train.userID.isin(u_id), :].to_csv(os.path.join(DATA_PATH, "add_FE_fixed_valid.csv"), index=False)
test.to_csv(os.path.join(DATA_PATH, "add_FE_fixed_test.csv"), index=False)


print(f"make fixed_data done")