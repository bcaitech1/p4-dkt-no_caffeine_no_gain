import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import pickle
from tqdm import tqdm

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

    def get_test_data(self):
        return self.test_data


    def sliding_window(self):
        args = self.args
        data = self.train_data
        
        window_size = args.max_seq_len
        stride = args.stride

        augmented_datas = []
        for row in data:
            seq_len = len(row[0])

            # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
            if seq_len <= window_size:
                augmented_datas.append(row)
            else:
                total_window = ((seq_len - window_size) // stride) + 1
                
                # 앞에서부터 slidding window 적용
                for window_i in range(total_window):
                    # window로 잘린 데이터를 모으는 리스트
                    window_data = []
                    for col in row:
                        window_data.append(col[window_i*stride:window_i*stride + window_size])

                    # Shuffle
                    # 마지막 데이터의 경우 shuffle을 하지 않는다
                    if args.shuffle and window_i + 1 != total_window:
                        shuffle_datas = self.shuffle(window_data, window_size, args)
                        augmented_datas += shuffle_datas
                    else:
                        augmented_datas.append(tuple(window_data))

                # slidding window에서 뒷부분이 누락될 경우 추가
                total_len = window_size + (stride * (total_window - 1))
                if seq_len != total_len:
                    window_data = []
                    for col in row:
                        window_data.append(col[-window_size:])
                    augmented_datas.append(tuple(window_data))

        return np.array(augmented_datas)

    def shuffle(self, data, data_size, args):
        shuffle_datas = []
        for i in range(args.shuffle_n):
            # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
            shuffle_data = []
            random_index = np.random.permutation(data_size)
            for col in data:
                shuffle_data.append(col[random_index])
            shuffle_datas.append(tuple(shuffle_data))
        return shuffle_datas


    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)


    def __preprocessing(self, main_df, sub_df=None, is_train=True):
        cate_cols = self.args.USE_COLUMN

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        all_df = pd.concat([main_df, sub_df])
        for col in cate_cols:            
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = all_df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                main_df[col] = main_df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            main_df[col]= main_df[col].astype(str)
            trans = le.transform(main_df[col])
            main_df[col] = trans
        
        return main_df

    def __feature_engineering(self, df):

        self.args.USERID_COLUMN = ['userID']
        self.args.USE_COLUMN = ['elapsed', 'time_bin', 'classification', 'paperNum', 'problemNum', 'user_total_acc', 'test_acc', 'assessment_acc', 'tag_acc', 'past_correct', 'past_content_count', 'correct_per_hour', 'same_tag', 'cont_tag']
        self.args.EXCLUDE_COLUMN = ['assessmentItemID', 'testId', 'Timestamp', 'KnowledgeTag',  'item', 'item_order', 'user_total_correct_cnt', 'user_total_ans_cnt',  'test_size', 'retest', 'user_test_ans_cnt', 'user_test_correct_cnt', 'user_acc', 'test_mean', 'test_sum', 'ItemID_mean', 'ItemID_sum','tag_mean', 'tag_sum', 'hours', 'time', 'correct_shift_-2', 'correct_shift_-1', 'correct_shift_1', 'correct_shift_2', 'total_used_time', 'shift',  'future_correct', 'past_content_correct', 'past_count', 'average_correct', 'average_content_correct', 'mean_time', 'time_median', 'hour',  'hour_mode', 'is_night', 'normalized_time', 'relative_time', 'time_cut', 'time_qcut']
        self.args.ANSWER_COLUMN = ['answerCode']
        
        assert df.head().shape[1] == len(self.args.USERID_COLUMN) + len(self.args.ANSWER_COLUMN) + len(
            self.args.USE_COLUMN) + len(self.args.EXCLUDE_COLUMN)

        return df

    def df_apply_function(self, r):
        return tuple([r[x].values for x in self.args.USE_COLUMN] + [r[x].values for x in self.args.ANSWER_COLUMN])

    def load_data_from_file(self, main_file_name, sub_file_name=None, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, main_file_name)
        main_df = pd.read_csv(csv_file_path)#, nrows=100000)
        
        # args.use_test_to_train이 True일때 test셋도 학습에 사용
        if self.args.use_test_to_train:
            csv_file_path = os.path.join(self.args.data_dir, self.args.test_file_name)
            test_df = pd.read_csv(csv_file_path)
            test_df = test_df[test_df.answerCode != -1].copy()
            main_df += test_df
            print("test셋 학습에 추가!")
            
        main_df = self.__feature_engineering(main_df)
        sub_df = None
        if is_train:
            csv_file_path = os.path.join(self.args.data_dir, sub_file_name)
            sub_df = pd.read_csv(csv_file_path)
            sub_df = self.__feature_engineering(sub_df)
            
        if self.args.model != 'tabnet':
            main_df = self.__preprocessing(main_df, sub_df, is_train)

            # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
            self.args.n_embedding_layers = []       # 나중에 사용할 떄 embedding key들을 저장
            for val in self.args.USE_COLUMN:
                self.args.n_embedding_layers.append(len(np.load(os.path.join(self.args.asset_dir, val+'_classes.npy'))))


        main_df = main_df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = self.args.USERID_COLUMN+self.args.USE_COLUMN+self.args.ANSWER_COLUMN
        group = main_df[columns].groupby('userID').apply(
                self.df_apply_function
            )
        if self.args.model =='tabnet':
            g = main_df[columns]
            return g
        return group.values        

    def load_train_data(self, train_file, valid_file):   
        self.train_data = self.load_data_from_file(train_file, valid_file)
        print("train data is loaded!")
        print()
            
        if self.args.window:
            self.train_data = self.sliding_window()
            
                
    def load_valid_data(self, valid_file):
        self.valid_data = self.load_data_from_file(valid_file, is_train=False)
        print("valid data is loaded!")
        print()
        

    def load_test_data(self, test_file):
        self.test_data = self.load_data_from_file(test_file, is_train=False)
        print("test data is loaded!")
        print()


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        cate_cols = [val for val in row]

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
    pin_memory = True
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