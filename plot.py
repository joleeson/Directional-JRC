import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, value="AverageCost", av_data = None):
    # if av_data is None:
    #     av_data = data
    
    sns.set(style="whitegrid", font_scale=1.1)
    myplot = sns.lineplot(x=data["Iteration"], y=data[value], hue=data[""])#, legend=False) #,data=data)
    
    """ Min/Max line and annotations """
    # x = np.array(data.groupby('').max()['Iteration'])
    # y = np.array(av_data.groupby('').max()[value])  # sorted in alphabetical order
    # text = 'max'
    # xy = (x/2,y)
    
    # plt.annotate(text+': %.1f'%y[1],            # Heuristic
    #               xy=(x[1]/2,y[1]),
    #               xytext = (0,8),# 10, 3 | 0,24
    #               textcoords="offset points",
    #               color='C1', #2',
    #               ha='center', va='bottom', size = 'smaller', weight = 'bold')
    # plt.annotate(text+': %.1f'%y[3],            # Round Robin
    #               xy=(x[3]/2,y[3]),
    #               xytext = (0,0),# 15, 3 | 0, 16
    #               textcoords="offset points",
    #               color='C0', #2',
    #               ha='center', va='bottom', size = 'smaller', weight = 'bold')
    # plt.annotate(text+': %.1f'%y[2],            # PPO
    #               xy=(x[2]/2,y[2]),
    #               xytext = (0,16),# 10, 3 | 0,32
    #               textcoords="offset points",
    #               color='C3', #3',
    #               ha='center', va='bottom', size = 'smaller', weight = 'bold')
    # plt.annotate(text+': %.1f'%y[0],            # A2C
    #               xy=(x[0]/2,y[0]),
    #               xytext = (0,0), # 3, 5 | 0,-24
    #               textcoords="offset points",
    #               color='C2', #'C4',
    #               ha='center', va='bottom', size = 'smaller', weight = 'bold')
    
    axes = myplot.axes
    axes.set_xlim(0,500)    # single agent
    plt.legend(bbox_to_anchor=(-0.17, 1.05), loc='lower left', borderaxespad=0.,ncol=4)     # single agent
    # plt.legend(bbox_to_anchor=(-0.1, 1.05), loc='lower left', borderaxespad=0.,ncol=2)      # multi-agent
    plt.show()
    

def av_data(data, values):
    
    num_seeds = data.max()['Unit']
    av_data = []
    values_ = values.copy()
    values_.append('Unit')
    data_values = pd.DataFrame(data, columns=values_)
    exp_names = pd.DataFrame(data, columns=[''])
    
    for i in range(num_seeds+1):
        if i == 0:
            av_data = pd.concat([(data_values[data_values['Unit'] == 0]), exp_names[data_values['Unit']==0]], axis=1)
        else:
            next_data = data_values[data_values['Unit'] == i]
            next_data.index = av_data.index[0:len(next_data.index)]
            # av_data[values_][0:len(next_data.index)] = av_data[values_][0:len(next_data.index)].add(next_data)
            data_alg = av_data['']
            av_data = av_data.add(next_data)
            av_data['']=data_alg
    
    av_data[values_] = av_data[values_].divide(num_seeds+1)
    
    return av_data

def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)
            
            experiment_data = experiment_data.ewm(span=10, adjust=False).mean()
            
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                '',#'Condition'
                condition or exp_name #[8:]
                )
            
            if (exp_name == 'test_env_ve'):
                experiment_data['Reward1'] = experiment_data['r_comm1'] - 8 * experiment_data['r_rad1']
            
            experiment_data['r_rad1'] = -experiment_data['r_rad1']
            
            experiment_data.rename(columns={'r_comm1': '$r_{comm} 1$',
                                            'r_rad1': '$r_{rad} 1$',
                                            'SINR_av': 'Average SIR',
                                            'SINR1': 'SNR 1',
                                            }
                                       ,inplace=True)
            if exp_name == 'test_v0e_unif_rand':
                experiment_data.rename(columns={'throughput1': 'Throughput1',
                                                'throughput_av': 'Average Throughput',
                                                }
                                       ,inplace=True)
            
            
            
            datasets.append(experiment_data)
            unit += 1
    
    return datasets


def main():
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument('logdir', nargs='*')
#    parser.add_argument('--legend', nargs='*')
#    parser.add_argument('--value', default='AverageReturn', nargs='*')
#    args = parser.parse_args()
    
    class DotDict(dict):
        def __init__(self, **kwds):
            self.update(kwds)
            self.__dict__ = self
    
    args = DotDict()
    
    args.logdir_rotation = (
        'data/test_env_ve_beamform_JRC_4lane-v0_23-09-2021_16-54-38_8usrs_rotate_wcomm11.0_w_rad1.0_maxage_8.0_ang35.0',    # ve
        )
    
    args.logdir_heuristic = (
        'data/test_env_ve_beamform_JRC_4lane-v0_23-09-2021_16-54-54_8usrs_heuristic_wcomm11.0_w_rad1.0_maxage_8.0_ang35.0', # ve THIS ONE
        )
    
    args.logdir_A2C = (
        'data/A2C2.6_env_ve_beamform_JRC_4lane-v0_27-09-2021_23-43-39_8users_wcomm11.0_w_rad8.0_maxage_8.0_ang35.0_obtimeTrue',
        )
    
    args.logdir_PPO = (
        'data/PPO2.6_env_ve_beamform_JRC_4lane-v0_27-09-2021_21-14-12_8users_wcomm11.0_w_rad8.0_maxage_8.0_ang35.0_obtimeTrue',
        )
        
    args.logdir = args.logdir_rotation + args.logdir_heuristic + args.logdir_A2C + args.logdir_PPO                          # single agent problem                                       # multi-agent problem
    
    """ Legend """
    args.legend = [r'Round Robin',r'Heuristic', r'A2C', r'PPO',]
    
    # args.value = ['Average Reward','Reward1','r_comm1','r_rad1','Average Throughput','Throughput1','SINR_av','SINR1']
    # args.value = ['Average Reward','Reward1','$r_{comm} 1$','$r_{rad} 1$','Average Throughput','Throughput1','Num Transmits 1','Average Num Transmits','Average SINR','SINR1']
    # args.value = ['Average Reward','Reward1','$r_{comm} 1$','$r_{rad} 1$','Throughput1','SNR 1']
    args.value = ['Reward1']
    # args.value = ['$r_{comm} 1$']
    # args.value = ['$r_{rad} 1$']
    # args.value = ['SNR 1']
    # args.value = ['Average Reward']
    # args.value = ['Average Reward','Reward1']
    # args.value = ['Entropy Bonus 1']
    # args.value = ['Average Reward', 'Reward1','Entropy Bonus 1']
    # args.value = ['Average Reward', 'Reward1','Entropy Bonus 1','radar action %','Throughput 1','comm action req %']
    # args.value = ['Average Reward', 'Reward1', 'Reward2', 'r_age', 'r_radar', 'r_overflow', 'throughput']
    # args.value = ['$r_{rad} 1$']
    
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True
        
    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)
            
    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        
    average_data = av_data(data, values)
    
    for value in values:
        # plot_data(data, value=value)
        plot_data(data, value=value, av_data = average_data)

if __name__ == "__main__":
    main()
