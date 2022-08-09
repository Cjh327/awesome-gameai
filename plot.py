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


def read_data(method, metric):
    data_n = []
    index = None
    min_length = float("inf")

    def dir_ls(dir_path):
        dir_list = os.listdir(dir_path)
        dir_list.sort()
        return dir_list

    data_dir = f'data/{method}/{metric}'

    print(data_dir)
    if not os.path.exists(data_dir):
        return None, None, None
    for filename in dir_ls(data_dir):
        if '.json' not in filename:
            continue

        print(os.path.join(data_dir, filename))

        with open(os.path.join(data_dir, filename), 'rb') as f:
            data = json.load(f)
            stats_values = data['stats_values'][0]
            stats_values = [x if x is not None else 1e-30 for x in stats_values]
            datetime_values = [datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in data['datetime']]
            t = [(d - datetime_values[0]).total_seconds() / 3600 for d in datetime_values]

            data_n.append(stats_values)
            if len(stats_values) < min_length:
                min_length = len(stats_values)
                index = t

        # if len(data_n[-1]) < min_length:
        #     min_length = len(data_n[-1])
        #     index = np.array(
        #         data)[:, 1] if data.shape[1] == 3 else np.array(data)[:, 0]

    data_n = [d[:min_length] for d in data_n]

    return np.array(data_n), min_length, np.array(index)


config = {
    '1': {
        'name': 'reward_sum',
        # draw_step: draw the points every draw_step step (should be in multiples of linear_interpolation_point_step)
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig

        'algs': ['Baseline', 'Hitted_1.5', 'Hitted_1.5-Sight_1.5', 'Hitted_1.5-Sight_1.5-Avoid_2',
                 'Hitted_1.5-Sight_2.5'],

        'metrics': ['0_reward_sum', '1_reward_sum', '2_reward_sum', '3_reward_sum'],

        'color': {
            # 'CollaQ': color_set['Amaranth'],
            # 'LINDA-QMIX': color_set['Amber'],
            'Hitted_1.5-Sight_2.5': color_set['orangered'],
            'Baseline': color_set['Electric violet'],
            'Hitted_1.5': color_set['Amber'],
            'Hitted_1.5-Sight_1.5': color_set['Dark sea green'],
            'Hitted_1.5-Sight_1.5-Avoid_2': color_set['Dark electric blue'],
            # 'RODE': color_set['Dark electric blue'],
            # 'LIAM': color_set['slategray']
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '2': {
        'name': 'coop',
        'draw_step': 2000,
        'y_label_step': 1,  # step of y label shown in the fig

        'algs': ['Baseline', 'Baseline-bt'],

        'metrics': ['oppo_team_final_res', 'self_team_winrate'],

        'color': {
            'Baseline-bt': color_set['Electric violet'],
            'Baseline': color_set['Amber'],
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '3': {
        'name': 'coop',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig

        'algs': ['Baseline-basemodel2', 'Baseline-nodar',
                 'basemodel2-HitThreat-0.1-KillThreat-2.4',
                 'nodar-HitThreat-0.1-KillThreat-2.4'],

        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', '0_reward_sum'],

        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'Baseline-nodar': color_set['lightblue'],
            'basemodel2-HitThreat-0.1-KillThreat-2.4': color_set['darkred'],
            'nodar-HitThreat-0.1-KillThreat-2.4': color_set['Electric violet'],
            # 'Baseline-nodar': color_set['Amaranth']
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '4': {
        'name': 'assist',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig

        'algs': ['Baseline-basemodel2', 'Assist'],

        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist'],

        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            # 'Baseline-nodar': color_set['lightblue'],
            # 'basemodel2-HitThreat-0.1-KillThreat-2.4': color_set['darkred'],
            'Assist': color_set['Electric violet'],
            # 'Baseline-nodar': color_set['Amaranth']
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '5': {
        'name': 'Assist_hp+1_w1',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['Baseline-basemodel2', 'Assist_hp+1_w1'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist'],
        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'Assist_hp+1_w1': color_set['Electric violet'],
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '6': {
        'name': 'Assist_latest15',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['Assist_latest15', 'Baseline-basemodel2', 'Coop_reward'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist'],
        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'Assist_latest15': color_set['Electric violet'],
            'Coop_reward': color_set['Sandy brown'],
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '7': {
        'name': 'Assist_attack&defend',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['Assist_Attack', 'Assist_Defend', 'Baseline-basemodel2', 'Coop_reward'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist'],
        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'Assist_Attack': color_set['Electric violet'],
            'Assist_Defend': color_set['cadetblue'],
            'Coop_reward': color_set['Sandy brown'],
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '8': {
        'name': 'HP',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['SigmoidHP', 'Baseline-basemodel2'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally'],
        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'SigmoidHP': color_set['Electric violet'],
            'Coop_reward': color_set['Sandy brown'],
        },
        'method_name': {
            'Baseline-basemodel2': 'Baseline',
            'SigmoidHP': '支援优化',
            'Coop_reward': '原优化方案'
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '9': {
        'name': 'Issue-orig-compare',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['Baseline-basemodel2', 'Coop_reward'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally'],
        'color': {
            'Baseline-basemodel2': color_set['orangered'],
            'Coop_reward': color_set['Electric violet'],
        },
        'method_name': {
            'Baseline-basemodel2': 'Baseline',
            'Coop_reward': '原优化方案'
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '10': {
        'name': 'Health-restore',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['Not_merged', 'Merged'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally',
                    'self_team_real_health_restore'],
        'color': {
            'Not_merged': color_set['orangered'],
            'Merged': color_set['Electric violet'],
        },
        'method_name': {
            'Not_merged': 'Not Merged',
            'Merged': 'Merged'
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '11': {
        'name': 'SeparateNetwork',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['shared', 'separate_dnn', 'separate_lstm_dnn', 'large_vnet', 'large_vnet_pnet'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist'],
        'color': {
            'shared': color_set['orangered'],
            'separate_dnn': color_set['Electric violet'],
            'separate_lstm_dnn': color_set['cadetblue'],
            'large_vnet': color_set['Sandy brown'],
            'large_vnet_pnet': color_set['Dark gray'],
        },
        'method_name': {
            'separate_dnn': 'Separate DNN',
            'separate_lstm_dnn': 'Separate LSTM and DNN',
            'large_vnet': 'Shared Network (larger value network)',
            'shared': 'Shared Network',
            'large_vnet_pnet': 'Shared Network (larger value network and policy network)'
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '12': {
        'name': 'PreRelease',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['pre_release', 'merge_health'],
        'metrics': ['shelter_health_restore'],
        # 'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally', 'real_health_restore', 'shelter_health_restore',
        # 'out_of_enemy_sight_health_restore', 'dying_health_restore_perc', 'health_restore_death_perc', 'dying_safe_health_restore_perc'],
        'compare': {
            'baseline': {'oppo_team_final_res': 9.15, 'self_team_winrate': 1.0, 'dist_to_ally': 33132.97,
                         'real_health_restore': 29.62, 'shelter_health_restore': 0.17,
                         'out_of_enemy_sight_health_restore': 0.65, 'dying_health_restore_perc': 0.03,
                         'dying_safe_health_restore_perc': 0.14, 'health_restore_death_perc': 0.03}
        },
        'color': {
            'pre_release': color_set['orangered'],
            'merge_health': color_set['Electric violet'],
            # 'separate_lstm_dnn': color_set['cadetblue'],
            # 'large_vnet': color_set['Sandy brown'],
            # 'large_vnet_pnet': color_set['Dark gray'],
            'baseline': 'black'
        },
        'method_name': {
            'pre_release': 'pre-release-1.19',
            'merge_health': 'Merge health restore',
            'baseline': 'Baseline'
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '13': {
        'name': 'lstm256',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['lstm256_shared_bt', 'lstm256_sepdnn_bt'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally', '0_reward_sum'],
        'color': {
            'lstm256_shared_bt': color_set['orangered'],
            'lstm256_sepdnn_bt': color_set['Electric violet'],
            # 'separate_lstm_dnn': color_set['cadetblue'],
            # 'large_vnet': color_set['Sandy brown'],
            # 'large_vnet_pnet': color_set['Dark gray'],
            # 'baseline': 'black'
        },
        'method_name': {
            'lstm256_shared_bt': 'Shared',
            'lstm256_sepdnn_bt': 'Separate DNN',
        },
        'smooth_range': 5,
        'data_scale': 1,
        'get_min_ratio': 0.25,
        'legend_prop_size': 20.0,
        'framealpha': 0.6,
        'legend_ncol': 1,
    },
    '14': {
        'name': 'lstm256_oppo',
        'draw_step': 2000,
        'draw_step_limit': 2e6,  # max step of x to draw, it is suggested to be in multiples of draw_step
        'y_label_step': 1,  # step of y label shown in the fig
        'algs': ['lstm256_shared_oppo', 'lstm256_sepdnn_oppo'],
        'metrics': ['oppo_team_final_res', 'self_team_winrate', 'self_team_assist', 'dist_to_ally', '0_reward_sum'],
        'color': {
            'lstm256_shared_oppo': color_set['orangered'],
            'lstm256_sepdnn_oppo': color_set['Electric violet'],
            # 'separate_lstm_dnn': color_set['cadetblue'],
            # 'large_vnet': color_set['Sandy brown'],
            # 'large_vnet_pnet': color_set['Dark gray'],
            # 'baseline': 'black'
        },
        'method_name': {
            'lstm256_shared_oppo': 'Shared DNN',
            'lstm256_sepdnn_oppo': 'Separate DNN',
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
    # plt.rc('font', family='Times New Roman')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib.rcParams['text.usetex'] = True
    plt.clf()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.xlim((0, 20))
    # plt.ylim((-80, 220))
    plt.tick_params('x', labelsize=24.0)
    plt.tick_params('y', labelsize=24.0)
    plt.xlabel('Time', {'size': 28.0})
    plt.ylabel('Value', {'size': 28.0})
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    def formatnum(x, pos):
        return str(x) + 'h'

    formatter = FuncFormatter(formatnum)
    ax.xaxis.set_major_formatter(formatter)

    return figure, ax


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--id', type=str, help='config id')

    args = parser.parse_args()

    conf = config[args.id]
    fig_dir = 'figures'
    if not os.path.exists(f'{fig_dir}'):
        os.mkdir(f'{fig_dir}')

    # load json data and calculate in the same time to save usage of memory for json_data
    index = None
    for metric in conf['metrics']:
        compare_index = np.zeros(1)
        figure, ax = init_figure()
        # load json
        for method in conf['algs']:
            data, length, index = read_data(method, metric)
            assert data is not None, metric + ' ' + method
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
                    label=conf['method_name'][method],
                    linewidth=3)
            if index[length - 1] > compare_index[-1]:
                compare_index = index[:length - 1]
        if 'compare' in conf:   
            for method in conf['compare']:
                if metric in conf['compare'][method]:
                    print(metric)
                    data_comp = np.ones_like(compare_index) * conf['compare'][method][metric]
                    ax.plot(compare_index, data_comp, color=conf['color'][method], label=conf['method_name'][method],
                            linewidth=3, linestyle='dashed')

        plt.legend(loc='best', prop={'size': conf['legend_prop_size']}, frameon=True,
                   framealpha=conf['framealpha'], facecolor='white', ncol=conf['legend_ncol'])
        plt.title(metric, fontsize=30)

        figure.savefig(f'{fig_dir}/{conf["name"]}-{metric}.png', bbox_inches='tight', dpi=600)
        # figure.savefig(f'{fig_dir}/{metric}.pdf', bbox_inches='tight', dpi=600)

        plt.show()
