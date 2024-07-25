import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

def load_npy(file_path):
    data_npy = np.load(file_path)
    data_pd = pd.DataFrame(data_npy)
    return data_pd

def preprocess_df(data, smoothing=1000, end=None):
    data.columns = ['Iteration', 'Constraint1_Limit', 'Constraint1', 'Constraint2_Limit', 'Constraint2']
    # data.columns = ['Iteration', 'Constraint1_Limit', 'Constraint1']
    data['Cumulative_Iteration'] = data['Iteration'].cumsum()
    if end is not None:
        min_num = 100000
        for i in range(len(data)):
            dif = end - data['Cumulative_Iteration'][i]
            if dif >= 0:
                if dif < min_num:
                    idx = i
                    min_num = dif
            else:
                break
        data = data[:idx]
        
    data['Constraint1_Limit_RollingMean'] = data['Constraint1_Limit'].rolling(window=smoothing).mean()
    data['Constraint1_Limit_RollingStd'] = data['Constraint1_Limit'].rolling(window=smoothing).std()

    data['Constraint1_RollingMean'] = data['Constraint1'].rolling(window=smoothing).mean()
    data['Constraint1_RollingStd'] = data['Constraint1'].rolling(window=smoothing).std()
    
    data['Constraint2_Limit_RollingMean'] = data['Constraint2_Limit'].rolling(window=smoothing).mean()
    data['Constraint2_Limit_RollingStd'] = data['Constraint2_Limit'].rolling(window=smoothing).std()
    
    data['Constraint2_RollingMean'] = data['Constraint2'].rolling(window=smoothing).mean()
    data['Constraint2_RollingStd'] = data['Constraint2'].rolling(window=smoothing).std()
    
    return data

def draw_plot(data1, constraint, label1="Desired Limit", label2="Constraint", figure_number=None, save_fig_path=None):
    font_size=15
    font_family='Ubuntu'
    plt.rc('font', family=font_family)
    
    if figure_number is not None:
        plt.figure(figure_number, figsize=(8, 6))
        ax = plt.gca()
    
    type=None
    if constraint == "Constraint1":
        title_name = " $\mathcal{S}_R$-Policy Episode Constraint"
        type="redundant"
    
        constraint_limit_type_mean = constraint +"_Limit_RollingMean"
        constraint_limit_type_std = constraint +"_Limit_RollingStd"
        ax.fill_between(data1['Iteration'],
                        data1[constraint_limit_type_mean] - data1[constraint_limit_type_std],
                        data1[constraint_limit_type_mean] + data1[constraint_limit_type_std],
                        color='g', alpha=0.1)
    
        constraint_type_mean = constraint +"_RollingMean"
        constraint_type_std = constraint +"_RollingStd"
        ax.fill_between(data1['Iteration'],
                        data1[constraint_type_mean] - data1[constraint_type_std],
                        data1[constraint_type_mean] + data1[constraint_type_std],
                        color='r', alpha=0.1)

        ax.plot(data1['Iteration'], np.full(len(data1['Iteration']), 0.2), label=label1, color='b', linestyle='--')
        ax.plot(data1['Iteration'], data1[constraint_limit_type_mean], label="Adaptive Constraint Threshold", color='g', linestyle='--')
        ax.plot(data1['Iteration'], data1[constraint_type_mean], label=label2, color='r')
    
    elif constraint == "Constraint2":
        title_name = " $\mathcal{S}_R$-Policy Episode Constraint"
        type="redundant"
    
        constraint_limit_type_mean = constraint +"_Limit_RollingMean"
        constraint_limit_type_std = constraint +"_Limit_RollingStd"
        ax.fill_between(data1['Iteration'],
                        data1[constraint_limit_type_mean] - data1[constraint_limit_type_std],
                        data1[constraint_limit_type_mean] + data1[constraint_limit_type_std],
                        color='g', alpha=0.1)
    
        constraint_type_mean = constraint +"_RollingMean"
        constraint_type_std = constraint +"_RollingStd"
        ax.fill_between(data1['Iteration'],
                        data1[constraint_type_mean] - data1[constraint_type_std],
                        data1[constraint_type_mean] + data1[constraint_type_std],
                        color='r', alpha=0.1)

        ax.plot(data1['Iteration'], np.full(len(data1['Iteration']), 0.5), label=label1, color='b', linestyle='--')
        ax.plot(data1['Iteration'], data1[constraint_limit_type_mean], label="Adaptive Constraint Threshold", color='g', linestyle='--')
        ax.plot(data1['Iteration'], data1[constraint_type_mean], label=label2, color='r')
        
    else:
        raise NameError
    
    ax.set_xlabel('iteration', fontsize=font_size)
    ax.set_ylabel('episode constraint', fontsize=font_size)

    # Set the font size for the ticks on the axes
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    

    # Add legend with custom font size
    formatter = FuncFormatter(lambda x, _: f'{int(x)}')
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(fontsize=font_size,loc='upper left')
    # ax.set_title(title_name, fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    
    if save_fig_path != None:
        plt.savefig(save_fig_path + f"{constraint}.png")
        plt.savefig(save_fig_path + f"{constraint}.svg")
    

smoothing = 1000

file_dir1 = "/home/kist/franka_walker/runs/adaptive_constraint/"
file_path1 = file_dir1 + "constraint.npy"
data1 = preprocess_df(load_npy(file_path1))
draw_plot(data1,"Constraint2", figure_number=0, save_fig_path=None)

plt.show()