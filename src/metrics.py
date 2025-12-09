import numpy as np
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    epsilon = 1e-8
    return np.mean(np.abs((pred - true) / (true + epsilon)))


def MSPE(pred, true):
    epsilon = 1e-8
    return np.mean(np.square((pred - true) / (true + epsilon)))

def sMAPE(pred, true):
    # 计算对称平均绝对百分比误差 (sMAPE)
    numerator = np.abs(true - pred)
    denominator = (np.abs(true) + np.abs(pred)) / 2
    smape = np.mean(numerator / (denominator + 1e-8)) * 100
    return smape

def torch_sMAPE(pred, true):
    numerator = torch.abs(true - pred)
    denominator = (torch.abs(true) + torch.abs(pred)) / 2
    smape = torch.mean(numerator / (denominator + 1e-8)) * 100
    return smape

def EVS(pred, true):
    var_true = np.var(true)
    var_pred = np.var(pred)
    return 1 - (var_pred / var_true)

def DTW(pred, true):
    # 计算动态时间规整 (DTW)
    dtw_distance, _ = fastdtw(true.reshape(-1,1), pred.reshape(-1,1), dist=euclidean)
    return dtw_distance

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = sMAPE(pred, true)
    evs = EVS(pred, true)
    dtw = DTW(pred, true)
    return mae, mse, rmse, mape, mspe, smape, evs, dtw


def tensor_metric(pred, true):
    mae = torch.mean(torch.abs(pred - true))
    mse = torch.mean((pred - true) ** 2)
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs((pred - true) / true))
    mspe = torch.mean(torch.square((pred - true) / true))
    smape = torch_sMAPE(pred, true)
    evs = 1 - (torch.var(pred) / torch.var(true))

    return mae, mse, rmse, mape, mspe, smape, evs