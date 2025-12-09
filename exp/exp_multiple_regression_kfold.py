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
import data_provider.data_loader_heiyi_kfold as data_loader_heiyi_kfold
import utils.plt_heiyi as plt_heiyi
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from exp.exp_basic import Exp_Basic
from utils.loss import WeightedMSELoss
import torch.profiler as profiler
warnings.filterwarnings('ignore')

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
        if args.load_checkpoint and os.path.exists(args.load_checkpoint):
            print(f"Loading checkpoint from: {args.load_checkpoint}")
            try:
                # 加载模型状态字典
                self.model.load_state_dict(torch.load(args.load_checkpoint, map_location=self.device))
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        elif args.load_checkpoint:
            print(f"Warning: Checkpoint file not found at {args.load_checkpoint}. Starting training from scratch.")

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
        self.args.size = [self.args.max_seq_len]
        if self.args.data_type == 'daily':
            if self.args.task_name == 'multiple_regression': # x->y
                if flag == 'train':
                    train_dataset = data_loader_heiyi_kfold.Dataset_regression_train_val(self.args)
                    return train_dataset
                elif flag == 'test':
                    test_dataset, test_loader = data_loader_heiyi_kfold.Dataset_regression_test(self.args)
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
            scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs // 2)
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

    def train(self, setting):
        train_data = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # summary(self.model, input_size=(self.args.batch_size, self.args.seq_len, self.args.enc_in))
        print_model_parameters(self.model)

        print('__________ Start training !____________')
        start_training_time = time.time()

        kfold = KFold(n_splits=self.args.num_fold)
        k_fold = kfold.split(range(len(train_data)))

        best_val_corrs = torch.tensor([], device=self.device)
        best_val_losses = torch.tensor([], device=self.device)
        for fold, (train_idx, val_idx) in enumerate(k_fold):
            start_fold_time = time.time()
            print(f"Training fold {fold + 1}/{self.args.num_fold}")
            # Subset train and validation data for the current fold
            train_subset = Subset(train_data, train_idx)
            val_subset = Subset(train_data, val_idx)

            train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=False, num_workers=10)
            vali_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                              drop_last=False, num_workers=10)
            
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
            best_val_corr = -1
            best_val_loss = 999

            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(f'training fold {fold+1}\n')

            for epoch in range(self.args.train_epochs):
                self.model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, time_gra, c_norm) in enumerate(train_loader):
                    self.args.c_norms = c_norm.float().to(self.device)
                    #if i == 0: print(batch_x.shape, batch_y.shape)
                    model_optim.zero_grad()
                    batch_x = [x.float().to(self.device) for x in batch_x] 
                    batch_y = batch_y.float().to(self.device)

                    if self.args.model == 'Path' or self.args.model == 'DUET':
                        outputs, moe_loss = self.model(batch_x)
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
                    if mask.all():
    
                        continue

                    if self.args.loss == 'MSE_with_weak':
                        tau_hat = torch.sigmoid(self.model.alpha)
                        tau = 1 - tau_hat
                        main_batch_x = batch_x[-1]
                        loss_dict = criterion(main_batch_x[~mask], outputs[~mask], batch_y[~mask], 
                                tau_hat, tau, self.args.c_norms)
                        #loss_dict = criterion(batch_x, outputs[~mask], batch_y[~mask], tau_hat, tau, self.args.c_norms)
                        mse = loss_dict['total']
                    else:
                        mse = criterion(outputs[~mask], batch_y[~mask])

                    if self.args.model == 'Path' or self.args.model == 'DUET':
                        loss = moe_loss + mse
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
                train_loss, corr, train_v_loss, train_p_loss, train_mse_loss = self.vali(train_subset, train_loader, criterion, fold) # 保持和val一致，每个epoch模型固定后train的corr
                vali_loss, vali_corr, vali_v_loss, vali_p_loss, vali_mse_loss = self.vali(val_subset, vali_loader, criterion, fold)
                mse_loss = train_loss # path模型还有moe loss，没有加在里面

                # 根据mse早停
                if vali_loss < best_val_loss:
                    best_epoch = epoch + 1
                    best_val_loss = vali_loss
                    best_val_corr = vali_corr
                    best_model_path = f'{path}/best_model_fold_{fold+1}.pth'
                    torch.save(self.model.state_dict(), best_model_path)

                print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | mse:{mse_loss:.8f} | Train Corr: {corr:.8f} "
                    f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f}")
                if self.args.loss == 'MSE_with_weak':
                    print(f"Epoch {epoch + 1} | Train v Loss: {train_v_loss:.7f} | Train p Loss:{train_p_loss:.8f}"
                        f"| Val v Loss: {vali_v_loss:.8f} | Val p Loss: {vali_p_loss:.8f}")
                    print(f"Epoch {epoch + 1} | tau hat: {torch.sigmoid(self.model.alpha).item():.7f}")

                with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Train Corr: {corr:.8f} "
                        f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f}\n")
                # Early stopping
                early_stopping(-vali_loss, self.model, path) # vali_corr vali_loss

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                if self.args.lradj != 'not':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            # torch.save(self.model.state_dict(), f'{path}/{fold+1}_last_epoch_model.pth')
            
            # Fold summary
            # print best vali loss
            print(f"Best validation loss for fold {fold+1}: {best_val_loss} at epoch {best_epoch}")
            best_val_losses = torch.cat([best_val_losses, best_val_loss.unsqueeze(0)])
            best_val_corrs = torch.cat([best_val_corrs, best_val_corr.unsqueeze(0)])
            fold_time = (time.time() - start_fold_time) / 60
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f'Best validation loss for fold {fold+1}: {best_val_loss} at epoch {best_epoch}\nfold{fold+1} training time: {fold_time:.2f} minutes\n')

        # Final training summary
        total_time = (time.time() - start_training_time) / 60
        with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write(f'Total training time: {total_time:.2f} minutes\nbest val loss:{torch.mean(best_val_losses)}\nbest_val corr:{torch.mean(best_val_corrs)}\n')
        print(f"Total training time: {total_time:.2f} minutes")
        print(f"best val loss: {torch.mean(best_val_losses)}")
        print(f"best val corr: {torch.mean(best_val_corrs)}")

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
            for i, (batch_x, batch_y, time_gra, c_norm) in enumerate(vali_loader):
                self.args.c_norms = c_norm.float().to(self.device)
                batch_x = [x.float().to(self.device) for x in batch_x]
                batch_y = batch_y.float().to(self.device)
                if self.args.model == 'AMD' or self.args.model == 'Path' or self.args.model == 'DUET':
                    outputs, _ = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.task_name == 'Long_term_forecasting':
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                outputs = outputs.squeeze(-1)
                batch_y = batch_y.squeeze(-1).squeeze(-1)
                mask = torch.zeros(batch_y.shape, dtype=torch.bool)
                if batch_y.isnan().sum() > 0:
                    mask = torch.isnan(batch_y)
                if mask.all():          
                    continue

                if self.args.loss == 'MSE_with_weak':
                    # mse = criterion(outputs[~mask], batch_y[~mask])
                    tau_hat = torch.sigmoid(self.model.alpha)
                    tau = 1 - tau_hat
                    main_batch_x = batch_x[-1]
                    loss_dict = criterion(main_batch_x[~mask], outputs[~mask], batch_y[~mask], 
                                tau_hat, tau, self.args.c_norms)

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
                
                nan_mask = torch.isnan(true)
                if nan_mask.any():
                    pred = pred[~nan_mask]
                    true = true[~nan_mask]

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
                for i, (batch_x, batch_y, time_gra, c_norm) in enumerate(test_loader):
                    # self.args.c_norms = c_norm.float().to(self.device)
                    batch_x = [x.float().to(self.device) for x in batch_x]
                    batch_y = batch_y.float().to(self.device)

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
                        # loss_dict = criterion(batch_x, outputs[~mask], batch_y[~mask], tau_hat, tau, self.args.c_norms)
                        # mse_loss.append(loss_dict['total'])
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

                # mse_loss_cpu = [loss.cpu().numpy() for loss in mse_loss]
                
                np.save(f'{self.args.save_path}/true',trues) 
                np.save(f'{self.args.save_path}/pred',preds) # 保存结果

                # mse = np.average(mse_loss_cpu)
                # print('test data mse: ', mse)
                
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
                plt_heiyi.plt_epoch_train_val_trend_fold(self.args, f'{self.args.save_path}/_result_of_multiple_regression.txt') # 画train过程图
        
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