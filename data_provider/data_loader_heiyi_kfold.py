import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
import pyarrow.feather as feather
import torch
import os
import pywt


class Dataset_regression():
    '''
    daily 原始特征
        - 数据处理：前值填充，robust
        - 不做截断，删异常值
    '''
    def __init__(self, args, data_path='/data/daily_label_research_v1_extend_factors.csv', flag='train',
                 size=None, train_start_year='2010',train_end_year='2018', test_year='2014',val_start_year='2019',ticker_type=0):
        if size == None:
            self.seq_len = 0
        else:
            self.seq_len = size[0]
        self.flag = flag
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year
        self.val_start_year = val_start_year
        self.test_year = test_year

        self.data_path = data_path
        self.ticker_type = ticker_type
        self.args = args
        self.pred_task = 'y'+str(self.args.pred_task)

        self.seq_len_list = args.seq_len_list
        self.max_seq_len = max(self.seq_len_list)

        self.__read_data__()

    
    def __get_data__(self):
        # df = pd.read_feather(self.data_path)
        df = pd.read_feather(self.data_path)

        df = df.iloc[:, :15]
        df = df.replace([-np.inf, np.inf], np.nan)
        columns = ['time', 'ticker', 'y10', 'main_ret_slp', 'ret', 'close_adj', 'high_adj', 'low_adj', 'open_adj',
                   'tr', 'volume', 'capvol0']
        feature_cols = ['main_ret_slp', 'ret', 'close_adj', 'high_adj', 'low_adj', 'open_adj', 'tr', 'volume',
                        'capvol0']
        df = df[columns]
        # 删异常值
        df.loc[df[self.pred_task] > 50, 'y10'] = np.nan
        #df = df[df[self.pred_task] <= 50].reset_index(drop=True)

        if self.data_path == '/data/stock_daily_2005_2021.feather':
            df['ret'] = df.pop('ret')

        df['time'] = pd.to_datetime(df['time'].astype(str), format='%Y%m%d')

        cols = ['main_ret_slp', 'close_adj', 'high_adj', 'low_adj', 'open_adj', 'tr', 'capvol0']
        train_set = pd.DataFrame()
        test_set = pd.DataFrame()
        grouped = df.groupby('ticker')
        for _, group in grouped:
            # 先做前值填充
            group['ret'] = group['ret'].fillna(0)
            group['volume'] = group['volume'].fillna(0)
            group[cols] = group[cols].ffill()
            # 删除第一行nan
            group.dropna(subset=group.columns[3:], inplace=True, how='any')

            data = group[
                (group['time'] >= str(self.train_start_year + '-01-01')) & (
                            group['time'] <= str(self.train_end_year + '-12-31'))]
            test = group[
                (group['time'] >= str(self.test_year + '-01-01')) & (group['time'] <= str(self.test_year + '-12-31'))]

            if len(data) <= self.args.pred_task + self.max_seq_len - 1:  # 当训练数据不足以生成完整的序列，这里多加了predtask
                if self.max_seq_len > 1:
                    test = pd.concat([data.iloc[-(self.max_seq_len - 1):], test])
                    if len(test) <=  self.max_seq_len - 1:  # 如果训练数据不足生成完整序列就打算把他放到test中生成新的test数据，但是需要防止新的test不能生成完整序列，这里如果加入了不完整的数据会导致加载数据错位，这个地方是不是不应该加predtask
                        continue
                    test_set = pd.concat([test_set, test])  # 短数据拼接到test，生成新的test数据
                else:
                    if len(test) <= self.max_seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                continue
            # 选取当前 ticker 的数据
            ticker_data = data.copy()
            if len(ticker_data.iloc[:-(self.args.pred_task)]) < self.max_seq_len:
                if len(test) <= self.max_seq_len - 1:
                    continue
                else:
                    if self.max_seq_len > 1:
                        test = pd.concat([data.iloc[-(self.max_seq_len - 1):], test])  # 为了预测test起始数据
                        test_set = pd.concat([test_set, test])
                    else:
                        test_set = pd.concat([test_set, test])
            else:
                train_data = ticker_data.iloc[:-(self.args.pred_task)]
                train_set = pd.concat([train_set, train_data])
                if self.max_seq_len > 1:
                    test = pd.concat([data.iloc[-(self.max_seq_len - 1):], test])
                    if len(test) <= self.max_seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                else:
                    if len(test) <= self.max_seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])

            
        # 对异常值做截断，排除y
        quantiles_label = train_set[self.pred_task].quantile([0.01, 0.99])
        # 提取分位数值（直接通过索引访问）
        q1 = quantiles_label[0.01]  # 0.01 分位数
        q99 = quantiles_label[0.99]  # 0.99 分位数
        # # 截断异常值
        
        train_set[self.pred_task] = np.clip(train_set[self.pred_task], q1, q99)
        train_set.loc[train_set[self.pred_task]>50,'y10']=np.nan
        quantiles = train_set[feature_cols].quantile([0.05, 0.95])
        quantiles = quantiles.transpose()
        quantiles.columns = ['q5', 'q95']
        for col in feature_cols:
            train_set[col] = np.clip(train_set[col], quantiles.loc[col, 'q5'], quantiles.loc[col, 'q95'])

        """
        params = np.load('/cpfs/fs3200/ppl/cta/project/flap01/submitted_code/e2eDig/ctac2_v2/code/model/model_parameter/p7_comb_v1/price/patchtst_gc_1_0/2024/robust_scaler.npz')
        scaler = RobustScaler(quantile_range=(5, 95))
        scaler.center_ = params['center_']
        scaler.scale_ = params['scale_']
        scaler.feature_names_in_ = params['feature_names_in_']
        scaler.n_features_in_ = params['n_features_in_']
        """
        
        scaler = RobustScaler(quantile_range=(5, 95))
        train_set[feature_cols] = scaler.fit_transform(train_set[feature_cols])
        
        test_set[feature_cols] = scaler.transform(test_set[feature_cols])
        joblib.dump(scaler, f'{self.args.save_path}/robust_scaler.pkl')

        self.args.c_norms = []
        # for c in range(3, 3+self.args.enc_in): # 按特征
        #     self.args.c_norms.append(np.linalg.norm(train_set.iloc[:, c],ord=2))
        # for c in f_values.columns:
        #     self.args.c_norms.append(f_values[c])
        return train_set, test_set


    def __read_data__(self):
        train_data,test_data = self.__get_data__()
        '''
        df_raw.columns: ['tk', 'y_5', 'y_20',...(features), ...(time_mark)
        '''
        if self.set_type == 0:
            self.train_ticker = train_data.iloc[:, 1:2]
            self.train_stamp = train_data.iloc[:,0:1]
            self.train_set = torch.tensor(train_data.iloc[:,3:].values)
            #self.train_f = torch.tensor(train_data.iloc[:,:].values)
            if self.args.pred_task==10:
                self.train_label = torch.tensor(train_data['y10'].values) # y10
            elif self.args.pred_task==20:
                self.train_label = torch.tensor(train_data['y20'].values) # y20
            elif self.args.pred_task==5:
                self.train_label = torch.tensor(train_data['y5'].values) # y5
        else:
            self.test_ticker = test_data.iloc[:,1:2]
            self.test_stamp = test_data.iloc[:,0:1]
            self.test_set = torch.tensor(test_data.iloc[:,3:].values)
            #self.test_f = torch.tensor(test_data.iloc[:,-self.n:].values)
            if self.args.pred_task==10:
                self.test_label = torch.tensor(test_data['y10'].values)
            elif self.args.pred_task==20:
                self.test_label = torch.tensor(test_data['y20'].values)
            elif self.args.pred_task==5:
                self.test_label = torch.tensor(test_data['y5'].values)

class Dataset_regression_dataset(Dataset):
    def __init__(self, data_x, data_y, tickers, data_stamp, seq_len_list):
        self.data_x = data_x
        self.data_y = data_y
        self.tickers = tickers
        self.data_stamp = data_stamp
        self.seq_len_list = seq_len_list
        self.max_seq_len = max(seq_len_list)


        # 创建 ticker 与其在数据中索引的映射
        self.start_indices=[]
        self.ticker_indices = {ticker: [] for ticker in self.tickers['ticker'].unique()}
        for i, ticker in enumerate(self.tickers['ticker']):
            self.ticker_indices[ticker].append(i)
        for i, (ticker, indices) in enumerate(self.ticker_indices.items()):
            self.start_indices.append(indices[0])

        self.total_windows = 0
        self.data_length = [0]
        for ticker, indices in self.ticker_indices.items():
            num_windows = len(indices) - self.max_seq_len + 1  # 计算有效滑动窗口
            if num_windows > 0:
                self.total_windows += num_windows
                self.data_length.append(self.total_windows)

    def __getitem__(self, index):
        
        time_gra = [0, 0, 0, 0, 1]
        for i in range(len(self.data_length)):
            if index < self.data_length[i]:
                current_idx_in_ticker = index - self.data_length[i - 1] 
                # 计算最长窗口的起始点，这是所有窗口共同的结束点基准
                s_end = self.start_indices[i - 1] + current_idx_in_ticker + self.max_seq_len
      
                seq_x_list = []
                for slen in self.seq_len_list:
                    curr_s_begin = s_end - slen
                    seq_x_sub = self.data_x[curr_s_begin:s_end]
                    seq_x_list.append(seq_x_sub)
                
                seq_y = self.data_y[s_end-1]

                time_gra = {'ticker': str(self.tickers.iloc[s_end-1:s_end].values[0]), 'time': str(self.data_stamp['time'].iloc[s_end-1:s_end].values[0])}

            
                max_s_begin = s_end - self.max_seq_len
                seq_f = self.data_y[max_s_begin:s_end]

                return seq_x_list, seq_y, time_gra, seq_f

    def __len__(self):
        return self.total_windows


def Dataset_regression_train_val(args):
    dataset = Dataset_regression(args, data_path=args.data_path,
                                 flag='train', size=args.size,
                                 train_start_year=args.train_start_year,
                                 train_end_year=args.train_end_year,
                                 val_start_year=args.val_start_year,
                                 test_year=args.test_year,
                                 ticker_type = args.ticker_type
                                 )
    seq_len_list = args.seq_len_list
    train_dataset = Dataset_regression_dataset(dataset.train_set, dataset.train_label, dataset.train_ticker, dataset.train_stamp,seq_len_list)
    return train_dataset


def Dataset_regression_test(args):
    dataset = Dataset_regression(args, data_path=args.data_path,
                                 flag='test', size=args.size,
                                 train_start_year=args.train_start_year,
                                 train_end_year=args.train_end_year,
                                 val_start_year=args.val_start_year,
                                 test_year=args.test_year,
                                 ticker_type=args.ticker_type
                                 )
    seq_len_list = args.seq_len_list
    test_dataset = Dataset_regression_dataset(dataset.test_set, dataset.test_label, dataset.test_ticker, dataset.test_stamp,seq_len_list)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, pin_memory=True,
                             drop_last=False, num_workers=10)
    return test_dataset, test_loader


