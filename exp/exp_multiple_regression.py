from src.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
from src.metrics import metric
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import data_provider.data_loader_heiyi_kfold as data_loader_heiyi_daily
import utils.plt_heiyi as plt_heiyi
from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
import seaborn as sns
from torch.utils.data import Dataset
from utils.loss import WeightedMSELoss
warnings.filterwarnings('ignore')

class TDataset(Dataset):
    def __init__(self, X, y, time_gra):        
        self.X = X 
        self.time_gra = time_gra
        self.y = y

    def __len__(self):
        return len(self.X[0]) 

    def __getitem__(self, idx: int):

        x_sample = [x[idx] for x in self.X] 
        return x_sample, self.y[idx], self.time_gra[idx]
    


    
def print_model_parameters(model, only_num=True):
    print('*************************Model Total Parameter************************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('************************Finish Parameter************************')

class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, y_true, y_pred):
        loss = 1 - self.cos(y_pred - y_pred.mean(dim=0, keepdim=True), y_true - y_true.mean(dim=0, keepdim=True))
        return loss  # 返回 1 减去 CCC 的值作为损失函数

class Exp_Multiple_Regression_Fold(Exp_Basic):
    def __init__(self, args):
        super(Exp_Multiple_Regression_Fold, self).__init__(args)
        self.all_test_preds = np.array([])

    def _build_model(self):
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        return self.model

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        self.args.size = [self.args.seq_len]
        if self.args.data_type == 'daily':
            if flag == 'train':
                train_dataset = data_loader_heiyi_daily.Dataset_regression_train_val(self.args) 
                return train_dataset
            elif flag == 'test':
                test_dataset, test_loader = data_loader_heiyi_daily.Dataset_regression_test(self.args)
                return test_dataset, test_loader

    def _select_optimizer(self):
        optim_type = self.args.optim_type
        if optim_type == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                    weight_decay=self.args.weight_decay)
        elif optim_type == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            raise ValueError("can't find your optimizer! please defined a optimizer!")
        scheduler = None
        if self.args.lradj == 'cos':
            scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs // 5)
        elif self.args.lradj == 'steplr':
            scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.5)
        return model_optim, scheduler

    def _select_criterion(self):
        loss_func = self.args.loss
        if loss_func == 'MSE':
            criterion = nn.MSELoss()
        elif loss_func == 'MAE':
            criterion = nn.L1Loss()
        elif loss_func == 'SmoothL1Loss':
            criterion = nn.SmoothL1Loss() 
        elif loss_func == 'ccc':
            criterion = CCC()
        elif loss_func == 'MSE_with_weak':
            criterion = WeightedMSELoss()
        else:
            raise ValueError("can't find your loss function! please defined it!")
        return criterion

    def prepared_dataset(self, train_data):
        # 初始化 data_x 为一个列表，长度等于 seq_len_list 的长度
        # 假设 seq_len_list = [120, 90, 60]，则 data_x = [[], [], []]
        num_scales = len(self.args.seq_len_list)
        data_x = [[] for _ in range(num_scales)] 
        data_y = []
        dates = []
        tickers = []
        
        temp_loader = DataLoader(dataset=train_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=10)
        
        for i, batch_data in enumerate(temp_loader):
            batch_x_list = batch_data[0] # 这是一个列表，包含 [Tensor(B, 120, D), Tensor(B, 90, D)...]
            batch_y = batch_data[1]
            time_gra = batch_data[2]
            
            # 分别将不同长度的数据存入对应的列表
            for scale_idx in range(num_scales):
                data_x[scale_idx].append(batch_x_list[scale_idx])

            # 处理 Y
            if not isinstance(batch_y, torch.Tensor):
                 batch_y = torch.tensor(batch_y)
            data_y.append(batch_y)

            # 处理 Time/Ticker (保持原样)
            if isinstance(time_gra, dict):
                d = [datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d%H:%M:%S') 
                     if isinstance(t, str) else t 
                     for t in time_gra['time']]
                ticker = [t.strip("[]' ") for t in time_gra['ticker']]
                dates.append(d)
                tickers.append(ticker)

        # 在循环结束后，对每个尺度分别进行 cat
        # 最终 data_x 是一个列表: [Tensor(Total, 120, D), Tensor(Total, 90, D), ...]
        final_data_x = []
        for scale_idx in range(num_scales):
            final_data_x.append(torch.cat(data_x[scale_idx], dim=0))
            
        final_data_y = torch.cat(data_y, dim=0)

        if dates:
            dates = np.concatenate(dates)
            tickers = np.concatenate(tickers)
            unique_dates = np.unique(dates)
            sorted_dates = np.sort(unique_dates)
            num_dates = len(sorted_dates)
        else:
            dates = np.array([])
            num_dates = 0
            sorted_dates = np.array([])
            
        return num_dates, sorted_dates, final_data_x, final_data_y, dates
    
    def load_dataset(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        # 计算当前fold的验证集日期范围
        fold_size = num_dates // self.args.num_fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold != self.args.num_fold - 1 else num_dates
        val_dates = sorted_dates[start_idx:end_idx]
        # train_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx:]])
        if (fold+1) == 1:
            train_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx+self.args.pred_task:]])
        elif (fold+1) > 1 and (fold+1) < self.args.num_fold:
            train_dates = np.concatenate([sorted_dates[:start_idx-self.args.pred_task], sorted_dates[end_idx+self.args.pred_task:]])
        elif (fold+1) == self.args.num_fold:
            train_dates = np.concatenate([sorted_dates[:start_idx-self.args.pred_task], sorted_dates[end_idx:]])
        # 创建mask
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)
        # 分割数据
        train_set_x = [x[train_mask] for x in data_x]
        val_set_x = [x[val_mask] for x in data_x]
        train_set_y = data_y[train_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=10)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=10)
        return train_dataset, train_loader, val_dataset, vali_loader
    
    def load_dataset_time(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        val_size = int(0.2 * num_dates)
        each_train_size = (num_dates - val_size) // self.args.num_fold
        start_idx = fold * each_train_size # train idx
        end_idx = num_dates - val_size
        # 去掉val前面的
        val_dates = sorted_dates[end_idx+self.args.pred_task:]
        train_dates = sorted_dates[start_idx:end_idx]
        
        # 创建mask
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)

        # 分割数据
        train_set_x = [x[train_mask] for x in data_x]
        val_set_x = [x[val_mask] for x in data_x]
        train_set_y = data_y[train_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=10)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=10)
        return train_dataset, train_loader, val_dataset, vali_loader
    
    def load_dataset_last(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        # 计算当前fold的验证集日期范围 最后20%
        fold_size = int(0.2*num_dates)
        start_idx = 0
        end_idx = num_dates - fold_size # train idx
        train_dates = sorted_dates[start_idx:end_idx-self.args.pred_task]
        val_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx:]])
        # 创建mask
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        # 分割数据
        train_set_x = [x[train_mask] for x in data_x]
        val_set_x = [x[val_mask] for x in data_x]
        train_set_y = data_y[train_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]
        
        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=10)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=10)
        return train_dataset, train_loader, val_dataset, vali_loader
    
    def train(self, setting):
        train_data = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        val_ratio = 1.0 / self.args.num_fold
        print(f'val_ratio:{val_ratio}\n')
        print_model_parameters(self.model)

        num_dates, sorted_dates, data_x, data_y, dates = self.prepared_dataset(train_data)

        print('__________ Start training !____________')
        start_training_time = time.time()

        best_val_corrs = torch.tensor([], device=self.device)
        best_val_losses = torch.tensor([], device=self.device)
        for fold in range(self.args.num_fold):
            start_fold_time = time.time()
            print(f"Training fold {fold + 1}/{self.args.num_fold}")

            if self.args.num_fold == 1:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset_last(num_dates, fold, sorted_dates, data_x, data_y, dates)
            else:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset(num_dates, fold, sorted_dates, data_x, data_y,dates)

            # 后续训练和验证步骤
            train_steps = len(train_loader)

            self.model = self._build_model()  # 每个折叠重新初始化模型
            model_optim, scheduler = self._select_optimizer()  # 每次初始化模型后也要重新初始化优化器和调度器
            criterion = self._select_criterion()
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            best_epoch = 0
            best_val_loss = 999
            best_val_corr = -1
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(f'training fold {fold+1}\n')

            for epoch in range(self.args.train_epochs):
                self.model.train()
                epoch_time = time.time()
                
                for i, (batch_x, batch_y, time_gra) in enumerate(train_loader):
                    model_optim.zero_grad()

                    if isinstance(batch_x, list):
                        batch_x = [x.float().to(self.device, non_blocking=True) for x in batch_x]
                    else:
                        batch_x = batch_x.float().to(self.device, non_blocking=True)
                        
                    batch_y = batch_y.float().to(self.device, non_blocking=True)

                    #if i == 0: print(batch_x.shape, batch_y.shape)

                    if self.args.model == 'DUET' or self.args.model == 'Path':
                        outputs, importance = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)

                    if self.args.task_name == 'Long_term_forecasting':
                        f_dim = 1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim]

                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y.squeeze(-1)
                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)

                    mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                    if batch_y.isnan().sum() > 0:
                        mask = torch.isnan(batch_y)

                    if self.args.loss == 'MSE_with_weak':
                        tau_hat = torch.sigmoid(self.model.alpha)
                        tau = 1 - tau_hat
                        
                        if isinstance(batch_x, list):
                            batch_x_masked = [x[~mask] for x in batch_x]
                        else:
                            batch_x_masked = batch_x[~mask]

                        loss_dict = criterion(batch_x_masked, outputs[~mask], batch_y[~mask], tau_hat, tau, self.args.c_norms)
                        mse = loss_dict['total']

                    else:
                        mse = criterion(outputs[~mask], batch_y[~mask])
                    if self.args.model == 'DUET' or self.args.model == 'Path':
                        loss = mse + importance
                    else: 
                        loss = mse

                    corr = torch.corrcoef(torch.stack([outputs[~mask].reshape(-1), batch_y[~mask].reshape(-1)]))[0, 1]

                    if (i == 0) or ((i + 1) % 1000 == 0):
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | corr: {3:.8f}".format(i + 1, epoch + 1,
                                                                                                loss.item(), corr))

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                        if self.args.grad_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)  # 进行梯度裁剪
                        scheduler.step()

                # Epoch end statistics
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss, corr, train_v_loss, train_p_loss, train_mse_loss = self.vali(train_dataset, train_loader, criterion, fold) # 保持和val一致，每个epoch模型固定后train的corr
                vali_loss, vali_corr, vali_v_loss, vali_p_loss, vali_mse_loss = self.vali(val_dataset, vali_loader, criterion, fold)
                mse_loss = train_loss
                
                if vali_loss < best_val_loss: 
                    best_epoch = epoch + 1
                    best_val_loss = vali_loss
                    best_val_corr = vali_corr
                    best_model_path = f'{path}/best_model_fold_{fold+1}.pth'
                    torch.save(self.model.state_dict(), best_model_path)

                print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.12f} | mse:{mse_loss:.12f} | Train Corr: {corr:.12f} "
                    f"| Val Loss: {vali_loss:.12f} | Val Corr: {vali_corr:.12f}")
                
                if self.args.loss == 'MSE_with_weak':
                    print(f"Epoch {epoch + 1} | Train v Loss: {train_v_loss:.7f} | Train p Loss:{train_p_loss:.8f}"
                        f"| Val v Loss: {vali_v_loss:.8f} | Val p Loss: {vali_p_loss:.8f}")
                    print(f"Epoch {epoch + 1} | tau hat: {torch.sigmoid(self.model.alpha).item():.7f}")

                with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f"Epoch {epoch + 1} | Train Loss: {train_loss:.12f} | Train Corr: {corr:.12f} "
                        f"| Val Loss: {vali_loss:.12f} | Val Corr: {vali_corr:.12f}\n")
                
                # Early stopping
                early_stopping(-vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                if self.args.lradj != 'not':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)
            
                # torch.save(self.model.state_dict(), f'{path}/epoch_{epoch+1}_model.pth')
            
            # Fold summary
            print(f"Best validation loss for fold {fold+1}: {best_val_loss} at epoch {best_epoch}")
            best_val_losses = torch.cat([best_val_losses, best_val_loss.unsqueeze(0)])
            best_val_corrs = torch.cat([best_val_corrs, best_val_corr.unsqueeze(0)])
            fold_time = (time.time() - start_fold_time) / 60
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f'Best validation loss for fold {fold+1}: {best_val_loss} at epoch {best_epoch}\nfold{fold+1} training time: {fold_time:.2f} minutes\n')

        # Final training summary
        total_time = (time.time() - start_training_time) / 60
        with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write(f'Total training time: {total_time:.2f} minutes\nbest val loss:{torch.mean(best_val_losses)}\nbest val corr:{torch.mean(best_val_corrs)}\n')
        print(f"Total training time: {total_time:.2f} minutes")
        print(f"best val loss: {torch.mean(best_val_losses)}")

        # Load the best model after training
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def vali(self, vali_data, vali_loader, criterion, fold):
        total_loss_list = []
        self.model.eval()
        preds_list = []
        trues_list = []
        mse_loss_list = []
        v_loss_list = []
        p_loss_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y, time_gra) in enumerate(vali_loader):
                if isinstance(batch_x, list):
                    batch_x = [x.float().to(self.device, non_blocking=True) for x in batch_x]
                else:
                    batch_x = batch_x.float().to(self.device, non_blocking=True)
                        
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                    

                if self.args.model == 'DUET' or self.args.model == 'Path':
                    outputs, _ = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.task_name == 'Long_term_forecasting':
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                outputs = outputs.squeeze(-1)
                batch_y = batch_y.squeeze(-1)
                mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                if batch_y.isnan().sum() > 0:
                    mask = torch.isnan(batch_y)

                if self.args.loss == 'MSE_with_weak':
                    tau_hat = torch.sigmoid(self.model.alpha)
                    tau = 1 - tau_hat

                    if isinstance(batch_x, list):
                        batch_x_masked = [x[~mask] for x in batch_x]
                    else:
                        batch_x_masked = batch_x[~mask]

                    loss_dict = criterion(batch_x_masked, outputs[~mask], batch_y[~mask], tau_hat, tau, self.args.c_norms)
                    loss = loss_dict['total']
                    mse_loss = loss_dict['mse']
                    v_loss = loss_dict['V_loss']
                    p_loss = loss_dict['P_loss']
                else:
                    loss = criterion(outputs[~mask], batch_y[~mask])

                pred = outputs.detach()
                true = batch_y.detach()
                total_loss_list.append(torch.tensor([loss.item()]).to(self.device))
                
                if self.args.loss == 'MSE_with_weak':
                    mse_loss_list.append(torch.tensor([mse_loss.item()]).to(self.device))
                    v_loss_list.append(torch.tensor([v_loss.item()]).to(self.device))
                    p_loss_list.append(torch.tensor([p_loss.item()]).to(self.device))

                pred = pred.squeeze(-1)
                true = true.squeeze(-1)
                if self.args.task_name == 'Long_term_forecasting':
                    pred = torch.sum(pred, dim=1)
                    true = torch.sum(true, dim=1)

                preds_list.append(pred)
                trues_list.append(true)

        total_loss = torch.cat(total_loss_list)
        if self.args.loss == 'MSE_with_weak':
            p_loss = torch.cat(p_loss_list)
            v_loss = torch.cat(v_loss_list)
            mse_loss = torch.cat(mse_loss_list)

        preds = torch.cat(preds_list).to(self.device)
        trues = torch.cat(trues_list).to(self.device)

        total_loss = torch.mean(total_loss)
        if self.args.loss == 'MSE_with_weak':
            p_loss = torch.mean(p_loss)
            v_loss = torch.mean(v_loss)
            mse_loss = torch.mean(mse_loss)

        mask = torch.zeros(trues.shape, dtype=torch.bool)

        vali_corr = torch.corrcoef(torch.stack([preds[~mask].reshape(-1), trues[~mask].reshape(-1)]))[0, 1]

        self.model.train()
        if self.args.loss == 'MSE_with_weak':
            return total_loss, vali_corr, v_loss, p_loss, mse_loss
        return total_loss, vali_corr, None, None, None

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        for fold in range(self.args.num_fold):
            # print(f'cycle:{cycle}-epoch:{epoch}')
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + '/' + setting, f'best_model_fold_{fold+1}.pth')))
            # scalers = joblib.load(f'{self.args.save_path}/robust_scaler.pkl')
            criterion = self._select_criterion()

            preds = []
            trues = []
            y_tickers = np.array([], dtype=str)
            y_times = np.array([], dtype=str)

            mse_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, time_gra) in enumerate(test_loader):
                    if isinstance(batch_x, list):
                        batch_x = [x.float().to(self.device, non_blocking=True) for x in batch_x]
                    else:
                        batch_x = batch_x.float().to(self.device, non_blocking=True)
                        
                    batch_y = batch_y.float().to(self.device, non_blocking=True)
                    

                    if self.args.model == 'AMD' or self.args.model == 'Path' or self.args.model == 'DUET':
                        outputs, _ = self.model(batch_x)
                    else:
                        outputs = self.model(batch_x)
                        
                    if self.args.task_name == 'Long_term_forecasting':
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    outputs = outputs.squeeze(-1)
                    batch_y = batch_y.squeeze(-1)
                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)

                    mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                    if batch_y.isnan().sum() > 0:
                        mask = torch.isnan(batch_y)
                    
                    # outputs = outputs * scalers.scale_[0] + scalers.center_[0]
                    # batch_y = batch_y * scalers.scale_[0] + scalers.center_[0]
                    if self.args.loss == 'MSE_with_weak':
                        tau_hat = torch.sigmoid(self.model.alpha)
                        tau = 1 - tau_hat

                        if isinstance(batch_x, list):
                            batch_x_masked = [x[~mask] for x in batch_x]
                        else:
                            batch_x_masked = batch_x[~mask]

                        loss_dict = criterion(batch_x_masked, outputs[~mask], batch_y[~mask], tau_hat, tau, self.args.c_norms)
                        mse_loss.append(loss_dict['total'])

                    else:
                        mse_loss.append(criterion(outputs[~mask], batch_y[~mask]))


                    pred = outputs.detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()

                    preds = np.append(preds, pred)
                    trues = np.append(trues, true)

                    y_ticker = time_gra['ticker']
                    y_time = time_gra['time']
                    # 日期格式转换
                    y_time = [datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d%H:%M:%S') if isinstance(t, str) else t for t in y_time]

                    # 优化字符串处理
                    y_ticker = [t.strip("[]' ") for t in y_ticker]

                    # 扩展列表
                    y_tickers = np.concatenate([y_tickers, y_ticker])
                    y_times = np.concatenate([y_times, y_time]) 

                y_tickers = np.array(y_tickers)
                y_times = np.array(y_times)

                mse_loss_cpu = [loss.cpu().numpy() for loss in mse_loss]

                np.save(f'{self.args.save_path}/true',trues)
                np.save(f'{self.args.save_path}/pred',preds)

                mse = np.average(mse_loss_cpu)
                print('test data mse: ',mse)
                
                mask = np.isnan(trues)
                corr = np.corrcoef(preds[~mask], trues[~mask])[0, 1] # 所有test的corr（拼接完一起）1折的
                print('the  test corr result is {}'.format(corr))

                if self.all_test_preds.size == 0:
                    # 将 self.all_test_preds 初始化为二维数组
                    self.all_test_preds = preds.reshape(1, -1)
                else:
                    self.all_test_preds = np.concatenate((self.all_test_preds, preds.reshape(1,-1)))

                data = {'ticker':y_tickers,'date':y_times,'True Values': trues, 'Predicted Values': preds}
                df = pd.DataFrame(data)

                # Define the path where you want to store the CSV file
                csv_file_path = self.args.save_path + '/' +self.args.model+self.args.task_name+self.args.test_year+f'predicted_true_values_{fold+1}.csv'

                df.to_csv(csv_file_path, index=False, mode='w')
                print("True and predicted values have been saved to:", csv_file_path)

                mae, mse, rmse, mape, mspe, smape, evs, dtw = metric(pred=preds[~mask], true=trues[~mask])
                
                current_time = datetime.now().strftime('%Y%m%d%H%M%S')
                f = open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a')
                f.write(setting + "\n" + current_time + " ")
                f.write(': the corr valued {} ;'.format(
                    corr) + ' the results of total horizon mae {}, mse {}, rmse {}, mape {}, mspe {}, smape {}, evs {}, dtw {}'.format(mae, mse, rmse, mape, mspe, smape, evs, dtw))
                f.write('\n')
                f.write('\n')
                f.close()
                plt_heiyi.plt_epoch_train_val_trend_fold(self.args, f'{self.args.save_path}/_result_of_multiple_regression.txt')
        
        all_test_mean_preds = np.mean(self.all_test_preds, axis=0)

        df_data = {'ticker':y_tickers,'date':y_times, 'True Values': trues, 'mean Predicted Values': all_test_mean_preds}
        test_mean_csv_file_path = self.args.save_path + '/' +self.args.model+self.args.task_name+self.args.test_year+f'predicted_true_values_mean.csv'
        mean_df = pd.DataFrame(df_data)
        mean_df.to_csv(test_mean_csv_file_path, index=False)

        mask = np.isnan(trues)
        all_test_corr = np.corrcoef(all_test_mean_preds[~mask], trues[~mask])[0, 1]
        print(f'the average corr value of {all_test_corr}')
        with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write(f'the average corr value of {all_test_corr}\n\n')

        return

