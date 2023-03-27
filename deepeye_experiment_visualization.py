r"""°°°
# Experiment Results

Goal: Begin exploring the transformation space

####  Method
1. Defined transforms 
    - Jitter
    - Rotate
    - Magnitude Warp
2. Grid Search different parameters for each transform
    - Set minimum by eye-balling transform with zero affect on original data
    - set maximum by eye-balling indistinguishable original data after transform
°°°"""
# |%%--%%| <ipQEFSX9xj|ZylzDMu60r>

import pickle
from scipy import interpolate
from sklearn import metrics
import matplotlib.pyplot as plt
from Evaluation import evaluation
import numpy as np

Y_columns = {'subId': 0,
 'session': 1,
 'trialId': 2,
 'seqId': 3,
 'original_trial_length_before_padding': 4}

train_data = np.load('Data/train_data.npz')
test_data = np.load('Data/test_data.npz')
X_train = train_data['X_train']
Y_train = train_data['Y_train']
X_test = test_data['X_test']
Y_test = test_data['Y_test']

window_sizes = [1,5,10]
plt.rcParams["figure.figsize"] = (10,8)

with open("metric_dict.pkl", "rb") as f:
    metric_dicts = pickle.load(f)

# with open("metrics_dict_new.pkl", "rb") as f:
#     metric_dicts = pickle.load(f)

# with open("metrics_dict_old.pkl", "rb") as f:
#     metric_dicts_old = pickle.load(f)

def plot(to_evaluate):
    for window_size in window_sizes:
        # for i, k in enumerate(metric_dicts.items()):
        for i, (k, v) in enumerate(metric_dicts.items()):
            try:
                if k not in to_evaluate:
                    continue
            except Exception as e:
                continue
            metric_dict = v
            model_name = k
            if i == 0:
                plot_random = True
            else:
                plot_random = False
            window_size = str(window_size)
            evaluation.avg_fnr_fpr_curve(metric_dict[window_size]['fpr'], metric_dict[window_size]['tpr'], 
                    label = model_name, plot_random=plot_random,
                    title = 'Window_size: ' + window_size, plot_statistics = False,
                    plot_points = 1000, ncol=1,
                    plot_legend=False,
                    fontsize = 14, xscale = 'log',
                    setting = 'identification')
        # plt.show()
        plt.savefig("./detcurve.jpg")
        break
        
        
def plot_eer(to_evaluate, window_sizes_to_plot=None, plot_name='./eerexp.png'):
    if window_sizes_to_plot is None:
        window_sizes_to_plot = window_sizes

    fig, axs = plt.subplots(len(window_sizes_to_plot), 1, figsize=(16, 7 * len(window_sizes_to_plot)))
    try:
        is_subscriptable = axs[0]
    except:
        axs = [axs]
        
    for idx, window_size in enumerate(window_sizes_to_plot):
        ax = axs[idx]
        mean_eer = []
        std_eer = []
        sample_names = []
        for sample_name, samples in to_evaluate.items():
            eers = []
            for metric_name in samples:
                window_size = str(window_size)
                if metric_name in metric_dicts: 
                    v = metric_dicts[metric_name][window_size]
                else:
                    v = metric_dicts[str(metric_name)][window_size]
                eers.append(get_eer(v['fpr'], v['tpr']))
            mean_eer.append(np.mean(eers))
            std_eer.append(np.std(eers))
            sample_names.append(sample_name)
        x_pos = np.arange(len(sample_names))
        ax.bar(x_pos, mean_eer, yerr=std_eer, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_title(f'Experiments with window size: {window_size}')
        ax.set_ylabel('EER Value')
        ax.yaxis.grid(True)
        print(idx)
        if idx + 1 == len(window_sizes_to_plot):
            ax.set_xticks(x_pos, rotation=60)
            ax.set_xticklabels(sample_names, rotation=90)
        else:
            ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight')
        
                  
def get_eer(fprs, tprs, plot_points=1000):
    """
    Plot average roc curve from multiple fpr and tpr arrays of multiple cv-folds

    :param fprs: list of fpr arrays for the different folds
    :param tprs: list of tpr arrays for the different folds
    """
    
    eers = []
    for i, fpr in enumerate(fprs):
        for idx, fpr_val in enumerate(fpr):
            if fpr_val > 1 - tprs[i][idx]:
                eers.append(fpr_val)       
                break

    eer = np.mean(eers)
    return eer

# |%%--%%| <ZylzDMu60r|vocwf8cCvF>

# no_aug = [k for k in metric_dicts_old.keys() if len(k) >= 1 and 'run' in k[0][0]]
# warp_15_knots_2 = [k for k in metric_dicts_old.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and type(k[0][1]) == tuple and 2 == k[0][1][1]]
# warp_15_knots_4 = [k for k in metric_dicts_old.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and 0.15 == k[0][1]]
# warp_15_knots_8 = [k for k in metric_dicts_old.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and type(k[0][1]) == tuple and 8 == k[0][1][1]]

# warp_knots = {
#     f'Warp 0.15 w 4 knots': warp_15_knots_4,
#     f'Warp 0.15 w 2 knots': warp_15_knots_2,
#     f'Warp 0.15 w 8 knots': warp_15_knots_8,
# }
# warp_knots

#|%%--%%| <vocwf8cCvF|9yI3oE7Pk7>


[k for k in metric_dicts.keys()]

#|%%--%%| <9yI3oE7Pk7|XsmwHJKANL>


percentage_augmentation = 1
all_to_evaluate = {}

# and k[1][1] = percentage_augmentation
warp_knots = {}
warps_std = {}
rotations = {}
jitter_std = {}
for percentage_augmentation in [1, 3]:
    no_aug = [k for k in metric_dicts.keys() if 'no_aug' in k[0]]
    warp_15_knots_2 = [k for k in metric_dicts.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and type(k[0][1]) == tuple and 2 == k[0][1][1] and k[1][1] == percentage_augmentation]
    warp_15_knots_4 = [k for k in metric_dicts.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and 0.15 == k[0][1][0] and 4 == k[0][1][1] and k[1][1] == percentage_augmentation]
    warp_15_knots_8 = [k for k in metric_dicts.keys() if len(k) >= 1 and 'mag_warp' in k[0][0] and type(k[0][1]) == tuple and 8 == k[0][1][1] and k[1][1] == percentage_augmentation]

    warp_knots[percentage_augmentation] = {
        f'Warp 0.15 w 4 knots': warp_15_knots_4,
        f'Warp 0.15 w 2 knots': warp_15_knots_2,
        f'Warp 0.15 w 8 knots': warp_15_knots_8,
    }
    warp_knots

    rotate_05 = [k for k in metric_dicts.keys() if 'rotate' in k[0][0] and 0.5 == k[0][1] and k[1][1] == percentage_augmentation]
    rotate_1 = [k for k in metric_dicts.keys() if 'rotate' in k[0][0] and 1 == k[0][1] and k[1][1] == percentage_augmentation]
    rotate_5 = [k for k in metric_dicts.keys() if 'rotate' in k[0][0] and 5 == k[0][1] and k[1][1] == percentage_augmentation]
    rotate_180 = [k for k in metric_dicts.keys() if 'rotate' in k[0][0] and 180 == k[0][1] and k[1][1] == percentage_augmentation]

    rotations[percentage_augmentation] = {
        f'Max Rotate Angle: 0.5': rotate_05,
        f'Max Rotate Angle: 1': rotate_1,
        f'Max Rotate Angle: 5': rotate_5,
        f'Max Rotate Angle: 180': rotate_180,
    }


    warps_0001 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.001 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_0005 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.005 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_001 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.01 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_003 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.03 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_01 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.1 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_03 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 0.3 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_1 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 1.0 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_15 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 1.5 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_2 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 2.0 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_3 = [k for k in metric_dicts.keys() if 'mag_warp' in k[0][0] and 3.0 == k[0][1][0] and k[1][1] == percentage_augmentation]
    warps_std[percentage_augmentation] = {
        f'Mag Warp - Alpha 0.001': warps_0001,
        f'Mag Warp - Alpha 0.005': warps_0005,
        f'Mag Warp - Alpha 0.01': warps_001,
        f'Mag Warp - Alpha 0.03': warps_003,
        f'Mag Warp - Alpha 0.1': warps_01,
        f'Mag Warp - Alpha 0.3': warps_03,
        f'Mag Warp - Alpha 1.0': warps_1,
        f'Mag Warp - Alpha 1.5': warps_15,
        f'Mag Warp - Alpha 2.0': warps_2,
        f'Mag Warp - Alpha 3.0': warps_3,
    }
    warps_std

    jitter_00001 = [k for k in metric_dicts.keys() if 'jitter' in k[0][0] and 0.0001 == k[0][1] and k[1][1] == percentage_augmentation]
    jitter_0001 = [k for k in metric_dicts.keys() if 'jitter' in k[0][0] and 0.001 == k[0][1] and k[1][1] == percentage_augmentation]
    jitter_0005 = [k for k in metric_dicts.keys() if 'jitter' in k[0][0] and 0.005 == k[0][1] and k[1][1] == percentage_augmentation]
    jitter_001 = [k for k in metric_dicts.keys() if 'jitter' in k[0][0] and 0.01 == k[0][1] and k[1][1] == percentage_augmentation]
    jitter_std[percentage_augmentation] = {
        f'Jitter with std: 0.0001': jitter_00001,
        f'Jitter with std: 0.001': jitter_0001,
        f'Jitter with std: 0.005': jitter_0005,
        # f'Jitter with std: 0.01': jitter_001,
    }
    jitter_std

    all_to_evaluate[percentage_augmentation] = {'No aug': no_aug, **warp_knots[percentage_augmentation], **warps_std[percentage_augmentation], **rotations[percentage_augmentation], **jitter_std[percentage_augmentation]}
all_to_evaluate

# |%%--%%| <XsmwHJKANL|QQoTm6MRKw>

def calculate_eers_window_10(perc_data):
    all_eers_window_10 = {}
    window_size = 10
    for sample_name, samples in perc_data.items():
        eers = []
        for metric_name in samples:
            window_size = str(window_size)
            if metric_name in metric_dicts: 
                v = metric_dicts[metric_name][window_size]
            else:
                v = metric_dicts[str(metric_name)][window_size]
            eer = get_eer(v['fpr'], v['tpr'])
            eers.append(eer)
        eers.append(np.mean(eers))
        eers.append(np.std(eers))
        eers = [f"{e:.3}" for e in eers]
        all_eers_window_10[sample_name] = eers
    return all_eers_window_10

all_eers_window_10 = {}
for perc, perc_data in all_to_evaluate.items():
    # print(perc)
    all_eers_window_10[perc] = calculate_eers_window_10(perc_data)
__import__('pprint').pprint({ f"{k}_{a}": b[-2:] for k, v in all_eers_window_10.items() for a, b in v.items() if k != 7})

#|%%--%%| <QQoTm6MRKw|UEWZ8JfBzy>

for perc in [1, 3]:
    magwarp_knots_plot = {'No aug': no_aug, **warp_knots[perc]}
    magwarp_knots_plot

    plot_eer(magwarp_knots_plot, ['10'], f'result_images/magwarpknots_{perc}.jpg')

#|%%--%%| <UEWZ8JfBzy|2IIeLcv7wM>

for perc in [1, 3]:
    magwarp_std_plot = {'No aug': no_aug, **warps_std[perc]}

    plot_eer(magwarp_std_plot, ['10'], f'result_images/magwarpstd_{perc}.jpg')

#|%%--%%| <2IIeLcv7wM|IlSFJ296e3>

for perc in [1, 3]:
    jitter_to_plot = {'No aug': no_aug, **jitter_std[perc]}

    plot_eer(jitter_to_plot, ['10'], f'result_images/jitterexp_{perc}.jpg')


#|%%--%%| <IlSFJ296e3|Kme2zNq8jl>

for perc in [1, 3]:
    rotate_to_plot = {'No aug': no_aug, **rotations[perc]}

    plot_eer(rotate_to_plot, ['10'], f'result_images/rotateexp_{perc}.jpg')


#|%%--%%| <Kme2zNq8jl|8qV2Jbenjc>

metric_dicts_old.keys()

#|%%--%%| <8qV2Jbenjc|B5wNvHOEiW>

import pandas as pd

print(pd.DataFrame({k: v[-2:] for k, v in all_eers_window_10.items() if 'No aug' in k or 'Jitter' in k}).transpose().set_axis(["Mean", "Standard Deviation"], axis=1).to_latex())

print(pd.DataFrame({k: v[-2:] for k, v in all_eers_window_10.items() if 'No aug' in k or 'Alpha' in k}).transpose().set_axis(["Mean", "Standard Deviation"], axis=1).to_latex())

print(pd.DataFrame({k: v[-2:] for k, v in all_eers_window_10.items() if 'No aug' in k or 'Warp' in k}).transpose().set_axis(["Mean", "Standard Deviation"], axis=1).to_latex())

print(pd.DataFrame({k: v[-2:] for k, v in all_eers_window_10.items() if 'No aug' in k or 'Rotate' in k}).transpose().set_axis(["Mean", "Standard Deviation"], axis=1).to_latex())


