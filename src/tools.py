import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange
plt.switch_backend('agg')


def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if args.lradj == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs // 5)
    elif args.lradj == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    # elif args.lradj in ['cos', 'steplr']:
        # assert scheduler != None
        # scheduler.step()
        # lr_adjust = {epoch: scheduler.get_last_lr()[-1]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return scheduler

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    def reset(self):
        """
        重置早停状态。
        """
        if self.verbose:
            print("Resetting early stopping state.")
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def lora_superposition(model,loraB):
    # wte_lora_B = []
    # wte_lora_A = []
    # for key in loraB.keys():
    #     if 'lora_layer_wte.A' in key:
    #         wte_lora_A = loraB[key].clone()
    #         # model[key] *= 0
    #     if 'lora_layer_wte.B' in key:
    #         wte_lora_B = loraB[key].clone()
    #         # model[key] *= 0
    # wte_lora = wte_lora_B @ wte_lora_A * (0.5)
    # wte_lora = wte_lora.permute(1, 0)
    # model['transformer.wte.weight'] += wte_lora
    for layer in range(6):
        q_lora_B=[]
        q_lora_A=[]
        k_lora_B = []
        k_lora_A = []
        q_lora_B_model = []
        q_lora_A_model = []
        k_lora_B_model = []
        k_lora_A_model = []
        for key in loraB.keys():
            if 'transformer.h.' + str(layer) + '.mixer.lora_layer.0.A' in key:
                q_lora_A = loraB[key].clone()
                q_lora_A_model = model[key].clone()
            if 'transformer.h.' + str(layer) + '.mixer.lora_layer.0.B' in key:
                q_lora_B = loraB[key]
                q_lora_B_model = model[key]
            if 'transformer.h.' + str(layer) + '.mixer.lora_layer.1.A' in key:
                k_lora_A = loraB[key]
                k_lora_A_model = model[key]
            if 'transformer.h.' + str(layer) + '.mixer.lora_layer.1.B' in key:
                k_lora_B = loraB[key]
                k_lora_B_model = model[key]
            # if 'transformer.h.'+ str(layer) + '.ln_1.weight' in key:
            #     ln_1_weight = loraB[key]
            #     model[key] = (ln_1_weight + model[key])/2
            # if 'transformer.h.'+ str(layer) + '.ln_1.bias' in key:
            #     ln_1_bais = loraB[key]
            #     model[key] = (ln_1_bais + model[key]) / 2
            # if 'transformer.h.'+ str(layer) + '.ln_2.weight' in key:
            #     ln_2_weight = loraB[key]
            #     model[key] = (ln_2_weight + model[key])/2
            # if 'transformer.h.'+ str(layer) + '.ln_2.bias' in key:
            #     ln_2_bais = loraB[key]
            #     model[key] = (ln_2_bais + model[key])/2
        q_lora = q_lora_B @ q_lora_A*(0.5)
        k_lora = k_lora_B @ k_lora_A*(0.5)
        q_lora_model = q_lora_B_model @ q_lora_A_model*(0.5)
        k_lora_model = k_lora_B_model @ k_lora_A_model*(0.5)
        # v_lora = torch.zeros_like(q_lora).cuda()
        for key in model.keys():
            if 'transformer.h.' + str(layer) + '.mixer.Wqkv.weight' in key:
                wqkv = loraB[key].clone()
                wqkv = wqkv.permute(1,0)
                Wqkv = rearrange(wqkv,"... (three d)   ->... three d  ", three=3).permute(1,0,2)
                q,k,v = Wqkv.unbind(dim=0)
                sim_q = F.cosine_similarity(q_lora,q_lora_model,dim=0)
                sim_k = F.cosine_similarity(k_lora,k_lora_model,dim=0)
                q = q+sim_q*q_lora
                k = k+sim_k*k_lora
                w = torch.cat((q,k,v), dim=1)
                w = w.permute(1,0)
                model[key] = w
    # for key in model.keys():
    #     if '.mixer.lora_layer' in key:
    #         model[key] = torch.zeros_like(model[key]).cuda()
    # pre1_lora_B = []
    # pre1_lora_A = []
    # pre2_lora_B = []
    # pre2_lora_A = []
    # for key in loraB.keys():
    #     if 'head.lora_layer_pre1.A' in key:
    #         pre1_lora_A = loraB[key].clone()
    #         # model[key] *=0
    #     if 'head.lora_layer_pre1.B' in key:
    #         pre1_lora_B = loraB[key].clone()
    #         # model[key] *= 0
    #     if 'head.lora_layer_pre2.A' in key:
    #         pre2_lora_A = loraB[key].clone()
    #         # model[key] *= 0
    #     if 'head.lora_layer_pre2.B' in key:
    #         pre2_lora_B = loraB[key].clone()
    #         # model[key] *= 0
    # pre1_lora = pre1_lora_B@pre1_lora_A*(0.5)
    # pre1_lora = pre1_lora.permute(1,0)
    # pre2_lora = pre2_lora_B@pre2_lora_A*(0.5)
    # pre2_lora = pre2_lora.permute(1,0)
    # model['head.linear1.weight'] +=pre1_lora
    # model['head.linear.weight'] +=pre2_lora

    return model