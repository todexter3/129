import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def plt_each_ticker_corr(args, dir):
    # 读取 CSV 文件
    data = pd.read_csv(dir)  
    fn = os.path.split(dir)[-2]

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.bar(data['ticker'], data['correlation'], color='skyblue')
    plt.xlabel('Ticker')
    plt.ylabel('Correlation')
    plt.title('Correlation by Ticker, year=2019')
    plt.xticks(rotation=90)  # 旋转 x 轴标签以便更好地显示
    plt.grid(axis='y')

    plt.ylim(-1, 1)

    # 显示图表
    plt.tight_layout()
    plt.savefig(f'{fn}/each_ticker_corr.png')
    plt.close()

def plt_epoch_train_val_trend(args, dir):
    # file_path = '/home/dmz-ai/liruoling/heiy/results/y20/Dlinear/log.txt'
    with open(dir, 'r') as file:
        log_data = file.read()
    fn = os.path.split(dir)[-2]
    # Regular expressions to extract train and val corr values
    train_corr_pattern = r"(?<!Batch Average )Train Corr: ([\-\d\.]+)"
    val_corr_pattern = r"Val Corr: ([\-\d\.]+)"

    # Extract train and val correlations
    train_corrs = [float(x) for x in re.findall(train_corr_pattern, log_data)]
    val_corrs = [float(x) for x in re.findall(val_corr_pattern, log_data)]

    # Epochs
    train_epochs = list(range(1, 1 + len(train_corrs)))
    val_epochs = list(range(1, 1 + len(val_corrs)))

    # Plot the correlations
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_corrs, label="Train Corr")
    plt.plot(val_epochs, val_corrs, label="Val Corr")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.title("Train and Validation Correlation over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{fn}/epoch_train_val_trend.png')
    plt.close() # 增加关闭图形

def plt_epoch_train_val_trend_fold(args, dir):
    # Read the log file
    with open(dir, 'r') as file:
        log_data = file.read()

    # Extract the folder name for saving the plots
    fn = os.path.split(dir)[-2]

    # Regular expressions to extract fold data
    fold_pattern = r"training fold (\d+)"
    train_corr_pattern = r"(?<!Batch Average )Train Corr: ([\-\d\.]+)"
    val_corr_pattern = r"Val Corr: ([\-\d\.]+)"


    # Find all folds
    fold_matches = re.findall(fold_pattern, log_data)

    # Split log data by folds
    fold_data = re.split(fold_pattern, log_data)[1:]  # Split the log data at "training fold X"
    
    # Iterate over each fold's data
    for i in range(0, len(fold_data), 2):
        fold_num = fold_data[i]  # Get the fold number
        fold_log = fold_data[i+1]  # Get the corresponding fold log content

        # Extract train and validation correlations for the fold
        train_corrs = [float(x) for x in re.findall(train_corr_pattern, fold_log)]
        val_corrs = [float(x) for x in re.findall(val_corr_pattern, fold_log)]

        # Create a list of epoch numbers for the fold
        train_epochs = list(range(1, 1 + len(train_corrs)))
        val_epochs = list(range(1, 1 + len(val_corrs)))

        # Plot train and validation correlations over epochs for the current fold
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_corrs, label="Train Corr")
        plt.plot(val_epochs, val_corrs, label="Val Corr")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.title(f"Train and Validation Correlation for Fold {fold_num}")
        plt.legend()
        plt.grid(True)

        save_dir = os.path.join(fn, f'epoch_train_val_trend_fold_{fold_num}.png')
        plt.savefig(save_dir)
        plt.close()

def plt_epoch_train_val_trend_mt(args, dir):
    # Read the log file
    with open(dir, 'r') as file:
        log_data = file.read()

    # Extract the folder name for saving the plots
    fn = os.path.split(dir)[-2]

    # Regular expressions to extract fold data
    fold_pattern = r"training fold (\d+)"
    train_corr_pattern = r"(?<!Batch Average )Main Train Corr: ([\-\d\.]+)"
    val_corr_pattern = r"Main Val Corr: ([\-\d\.]+)"


    # Find all folds
    fold_matches = re.findall(fold_pattern, log_data)

    # Split log data by folds
    fold_data = re.split(fold_pattern, log_data)[1:]  # Split the log data at "training fold X"
    
    # Iterate over each fold's data
    for i in range(0, len(fold_data), 2):
        fold_num = fold_data[i]  # Get the fold number
        fold_log = fold_data[i+1]  # Get the corresponding fold log content

        # Extract train and validation correlations for the fold
        train_corrs = [float(x) for x in re.findall(train_corr_pattern, fold_log)]
        val_corrs = [float(x) for x in re.findall(val_corr_pattern, fold_log)]

        # Create a list of epoch numbers for the fold
        train_epochs = list(range(1, 1 + len(train_corrs)))
        val_epochs = list(range(1, 1 + len(val_corrs)))

        # Plot train and validation correlations over epochs for the current fold
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_corrs, label="Train Corr")
        plt.plot(val_epochs, val_corrs, label="Val Corr")
        plt.xlabel("Epoch")
        plt.ylabel("Correlation")
        plt.title(f"Train and Validation Correlation for Fold {fold_num}")
        plt.legend()
        plt.grid(True)

        # Save the plot for this fold
        save_dir = os.path.join(fn, f'epoch_train_val_trend_fold_{fold_num}.png')
        plt.savefig(save_dir)
        plt.close()

def plt_epoch_train_val_trend_500ms(args, dir):
    # Read the log file
    with open(dir, 'r') as file:
        log_data = file.read()

    # Extract the folder name for saving the plots
    fn = os.path.split(dir)[-2]

    # Regular expressions to extract fold data
    fold_pattern = r"training fold (\d+)"
    train_corr_pattern = r"Train Loss: ([\-\d\.]+)"
    val_corr_pattern = r"Val Loss: ([\-\d\.]+)"


    # Find all folds
    fold_matches = re.findall(fold_pattern, log_data)

    # Split log data by folds
    fold_data = re.split(fold_pattern, log_data)[1:]  # Split the log data at "training fold X"
    
    # Iterate over each fold's data
    for i in range(0, len(fold_data), 2):
        fold_num = fold_data[i]  # Get the fold number
        fold_log = fold_data[i+1]  # Get the corresponding fold log content

        # Extract train and validation correlations for the fold
        train_corrs = [float(x) for x in re.findall(train_corr_pattern, fold_log)]
        val_corrs = [float(x) for x in re.findall(val_corr_pattern, fold_log)]

        # Create a list of epoch numbers for the fold

        train_epochs = list(range(1, 1 + len(train_corrs)))
        val_epochs = list(range(1, 1 + len(val_corrs)))

        # Plot train and validation correlations over epochs for the current fold
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_corrs, label="Train loss")
        plt.plot(val_epochs, val_corrs, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.title(f"Train and Validation loss for Fold {fold_num}")
        plt.legend()
        plt.grid(True)

        # Save the plot for this fold
        save_dir = os.path.join(fn, f'epoch_train_val_trend_fold_{fold_num}.png')
        plt.savefig(save_dir)
        plt.close()  