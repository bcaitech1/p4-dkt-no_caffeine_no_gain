import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, valid_ratio=0.3, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        train_ratio = 1 - valid_ratio

        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * train_ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = ['KnowledgeTag', 'classification', 'paperNum', 'problemNum']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            
            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        df['classification'] = df['testId'].str[2:3]
        df['paperNum'] = df['testId'].str[-3:]
        df['problemNum'] = df['assessmentItemID'].str[-3:]

        df = df.astype({'Timestamp': 'datetime64[ns]'})
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
            return 999
        
        df["time_bin"] = df.hours.apply(time_bin)
        df = df.astype({'Timestamp': 'str'})
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        print(len(df.elapsed.unique()), len(df.time_bin.unique()))

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

                
        # self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        # self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_class = len(np.load(os.path.join(self.args.asset_dir, 'classification_classes.npy')))
        self.args.n_paper = len(np.load(os.path.join(self.args.asset_dir, 'paperNum_classes.npy')))
        self.args.n_problem = len(np.load(os.path.join(self.args.asset_dir, 'problemNum_classes.npy')))
        self.args.n_elapsed = 252
        self.args.n_time_bin = 5
        


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'KnowledgeTag', 'classification', 'paperNum', 'problemNum', 'elapsed', 'time_bin', 'answerCode']
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    # r['testId'].values, 
                    # r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['classification'].values,
                    r['paperNum'].values,
                    r['problemNum'].values,
                    r['elapsed'].values,
                    r['time_bin'].values,
                    r['answerCode'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        tag, classification, paper, problem, elapsed, time_bin, correct = row[0], row[1], row[2], row[3], row[4], row[5], row[6]

        cate_cols = [tag, classification, paper, problem, elapsed, time_bin, correct]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader