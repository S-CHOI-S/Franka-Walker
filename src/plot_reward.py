import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter

def load_npy(file_path):
    data_npy = np.load(file_path)
    data_pd = pd.DataFrame(data_npy)
    return data_pd

def preprocess_df(data, smoothing=1000, end=None):
    data.columns = ['Iteration', 'Reward']
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

    data['Reward_RollingMean'] = data['Reward'].rolling(window=smoothing).mean()
    data['Reward_RollingStd'] = data['Reward'].rolling(window=smoothing).std()
    
    return data

def draw_plot(data1, reward, label1="PPO", label2="RL", figure_number=None, save_fig_path=None):
    font_size=18
    font_family='Ubuntu'
    plt.rc('font', family=font_family)
    
    if figure_number is not None:
        plt.figure(figure_number, figsize=(8, 6))
        ax = plt.gca()
    
    type=None
    if reward == "Reward":
        title_name = " $\mathcal{S}_R$-Policy Episode Return"
        type="redundant"
    else:
        raise NameError
    
    reward_type_mean = reward +"_RollingMean"
    reward_type_std = reward +"_RollingStd"
    ax.fill_between(data1['Iteration'],
                    data1[reward_type_mean] - data1[reward_type_std],
                    data1[reward_type_mean] + data1[reward_type_std],
                    color='r', alpha=0.1)

    ax.plot(data1['Iteration'], data1[reward_type_mean], label=label1, color='r')
    ax.set_xlabel('iteration', fontsize=font_size)
    ax.set_ylabel('episode return', fontsize=font_size)

    # Set the font size for the ticks on the axes
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)

    # Add legend with custom font size
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # ax.xaxis.set_major_formatter(formatter)
    formatter = FuncFormatter(lambda x, _: f'{int(x)}')
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(fontsize=font_size,loc='upper left')
    # ax.set_title(title_name, fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    
    if save_fig_path != None:
        plt.savefig(save_fig_path + "reward.png")
        plt.savefig(save_fig_path + "reward.svg")
    
smoothing = 1000

file_dir1 ="/home/kist/franka_walker/runs/ves/" 
file_path1 = file_dir1 + "reward.npy"
data1 = preprocess_df(load_npy(file_path1))
draw_plot(data1,"Reward", figure_number=0, save_fig_path=None)

plt.show()
# 20240716_14-00-55
# 20240715_19-42-33