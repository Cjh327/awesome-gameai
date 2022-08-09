import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter

# linear_interpolation_point_step = 20000
# plt.switch_backend('agg')


# http://latexcolor.com/
color_set = {
    'Amaranth': np.array([0.9, 0.17, 0.31]),  # main algo
    'Amber': np.array([1.0, 0.49, 0.0]),  # main baseline
    'Bleu de France': np.array([0.19, 0.55, 0.91]),
    'Electric violet': np.array([0.56, 0.0, 1.0]),
    'Dark sea green': np.array([0.56, 0.74, 0.56]),
    'Dark electric blue': np.array([0.33, 0.41, 0.47]),
    'Dark gray': np.array([0.66, 0.66, 0.66]),
    'Arsenic': np.array([0.23, 0.27, 0.29]),
    # new
    'French beige': np.array([0.65, 0.48, 0.36]),
    'Grullo': np.array([0.66, 0.6, 0.53]),
    'Blush': np.array([0.87, 0.36, 0.51]),
    'Dark coral': np.array([0.8, 0.36, 0.27]),
    'Old lavender': np.array([0.47, 0.41, 0.47]),
    'Sandy brown': np.array([0.96, 0.64, 0.38]),
    'Dark cyan': np.array([0.0, 0.55, 0.55]),
    'Firebrick1': np.array([255., 48., 48.]) / 255,
    'Brown': np.array([255., 64., 64.]) / 255,
    # new 2
    'darkred': '#8B0000',
    'darkblue': '#00008B',
    'orangered': '#FF4500',
    'red': 'red',
    'blue': 'blue',
    'lightblue': '#ADD8E6',
    'dodgerblue': '#1E90FF',
    'indianred': '#CD5C5C',
    'mediumvioletred': '#C71585',
    'palevioletred': '#DB7093',
    'aliceblue': '#F0F8FF',
    'blueviolet': '#8A2BE2',
    'cadetblue': '#5F9EA0',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'Gray-asparagus': '#465945',
    'hotpink': '#FF69B4',
    'gold': '#FFD700',
    'orange': '#FFA500',
    'slategray': '#708090',
}


def smooth_single(data, smooth_range):
    # print('smooth', type(data), len(data))
    new_data = np.zeros_like(data)
    for i in range(0, data.shape[-1]):
        if i < smooth_range:
            new_data[i] = 1. * np.sum(data[:i + 1], axis=0) / (i + 1)
        else:
            new_data[i] = 1. * np.sum(data[i - smooth_range + 1:i + 1], axis=0) / smooth_range

    return new_data


def read_data(dir, method, env_name):
    data_n = []
    index = None
    min_length = 9999999

    def dir_ls(dir_path):
        dir_list = os.listdir(dir_path)
        dir_list.sort()
        return dir_list

    data_dir = os.path.join(dir, env_name, method)
    print(data_dir)
    
    if not os.path.exists(data_dir):
        return None, None, None
    
    data_n = []
    for filename in dir_ls(data_dir):
        if '.json' not in filename:
            continue

        print(os.path.join(data_dir, filename))

        with open(os.path.join(data_dir, filename), 'rb') as f:
            data = np.array(json.load(f))

            assert(data.shape[1] == 3)
            data_n.append(data[:, -1])
            index = data[:, -2]
            min_length = min(min_length, index.shape[0])

        data_n = [d[:min_length] for d in data_n]

    return np.array(data_n), min_length, np.array(index)


config = {
    '1': {
        'name': 'DT',
        # draw_step: draw the points every draw_step step (should be in multiples of linear_interpolation_point_step)
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig

        'algs': ['large', 'atari', 'small'],
        
        'compare': {
            'dreamer': {'dmc_cheetah_run': 777.8, 'dmc_pendulum_swingup': 833.5, 'dmc_reacher_easy': 972.3, 'dmc_quadruped_walk': 937.6, 'dmc_cartpole_swingup': 846.3},
        },
        
        'alg_name': {
            'small': 'DT-small (0.6M)',
            'atari': 'DT-atari (2.4M)',
            'large': 'DT-large (16.3M)',
            'dreamer': 'Dreamer',
        },

        'color': {
            'small': color_set['Dark electric blue'],
            'atari': color_set['Amber'],
            'large': color_set['Amaranth'],
            'dreamer': 'black',
            # 'CollaQ': color_set['Amaranth'],
            # 'LINDA-QMIX': color_set['Amber'],
            # 'RODE': color_set['Dark electric blue'],
            # 'LIAM': color_set['slategray']
            # 'Baseline': color_set['Electric violet'],
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
}


def init_figure():
    figure = plt.figure(figsize=(7 * 2, 4.8 * 2))
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', family='Times New Roman')
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib.rcParams['text.usetex'] = True
    plt.clf()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # plt.xlim((0, 20))
    # plt.ylim((-80, 220))
    plt.tick_params('x', labelsize=24.0)
    plt.tick_params('y', labelsize=24.0)
    plt.xlabel('Epochs', {'size': 28.0})
    plt.ylabel('Return', {'size': 28.0})
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    def formatnum(x, pos):
        return str(x / 1000) + 'K'

    formatter = FuncFormatter(formatnum)
    ax.xaxis.set_major_formatter(formatter)

    return figure, ax


if __name__ == '__main__':
    

    parser = ArgumentParser()

    parser.add_argument('--id', type=str, default='1', help='config id')
    parser.add_argument('--fig-dir', type=str, default='figures', help='figure directory')

    args = parser.parse_args()

    conf = config[args.id]
    if not os.path.exists(f'{args.fig_dir}'):
        os.mkdir(f'{args.fig_dir}')

    # load json data and calculate in the same time to save usage of memory for json_data
    index = None
    for env_name in ['dmc_cheetah_run', 'dmc_cartpole_swingup', 'dmc_quadruped_walk']:
        compare_index = np.zeros(1)
        figure, ax = init_figure()
        # load json
        for method in conf['algs']:
            data, length, index = read_data('json_data', method, env_name)
            assert data is not None, env_name + ' ' + method
            data_median = np.median(data, axis=0) * conf['data_scale']
            data_sort = np.sort(data, axis=0) * conf['data_scale']

            num_seed = data.shape[0]
            num_get_min = int(num_seed * conf['get_min_ratio'])
            num_get_max = -num_get_min - 1
            data_min = data_sort[num_get_min, :]
            data_max = data_sort[num_get_max, :]
            data_median = smooth_single(data_median, conf['smooth_range'])
            data_min = smooth_single(data_min, conf['smooth_range'])
            data_max = smooth_single(data_max, conf['smooth_range'])
            ax.fill_between(index[:length], data_min, data_max, alpha=0.1, color=conf['color'][method],
                            linewidth=0)
            ax.plot(index[:length - 1], data_median[:length - 1], color=conf['color'][method],
                    label=conf['alg_name'][method],
                    linewidth=3)
            if index[length - 1] > compare_index[-1]:
                compare_index = index[:length - 1]
        if 'compare' in conf:   
            for method in conf['compare']:
                data_comp = np.ones_like(compare_index) * conf['compare'][method][env_name]
                ax.plot(compare_index, data_comp, color=conf['color'][method], label=conf['alg_name'][method],
                        linewidth=3, linestyle='dashed')

        plt.legend(loc='best', prop={'size': conf['legend_prop_size']}, frameon=True,
                   framealpha=conf['framealpha'], facecolor='white', ncol=conf['legend_ncol'])
        plt.title(env_name, fontsize=30)

        figure.savefig(f'{args.fig_dir}/{conf["name"]}-{env_name}.png', bbox_inches='tight', dpi=600)

        # plt.show()
