"""°°°
## Data Enhancement and Augmentation Techniques for Oculomotoric Biometric Identification using DeepEye
°°°"""
# |%%--%%| <Qt8bzY9XhR|eagLlzfCBH>

import os
import socket

%pylab inline
%load_ext autoreload
%autoreload 2
print(socket.gethostname())

# |%%--%%| <eagLlzfCBH|qSVyi9sIQL>

import joblib
import numpy as np
import random
from Evaluation import evaluation
import sys
import seaborn as sns
from tqdm.notebook import tqdm

# |%%--%%| <qSVyi9sIQL|3NRBkmOEaI>
"""°°°
## Download the data
* the data can be found here: 
    * Data/test_data.npz https://osf.io/g8rvb/download
    * Data/train_data.npz https://osf.io/s7vay/download
* the file structure should look like this:
    * Data/
    * ├── test_data.npz
    * └── train_data.npz
°°°"""
# |%%--%%| <3NRBkmOEaI|RnrAQgBQZl>

if not os.path.exists('Data/'):
    os.makedirs('Data/')
if not os.path.exists('Data/test_data.npz'):
    !wget -O Data/test_data.npz https://osf.io/g8rvb/download
if not os.path.exists('Data/train_data.npz'):
    !wget -O Data/train_data.npz https://osf.io/s7vay/download
        
if not os.path.exists('trained_models/'):
    os.makedirs('trained_models/')

# |%%--%%| <RnrAQgBQZl|7g1usC6k0w>
"""°°°
## Set up the GPU you want to train on
* if you want to train on Google-Colab or the CPU you don't need to specify the GPU
°°°"""
# |%%--%%| <7g1usC6k0w|4gAublfzpQ>

flag_train_on_gpu = True
GPU = 0
if flag_train_on_gpu:
    import tensorflow as tf
    # select graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)

# |%%--%%| <4gAublfzpQ|J9uLvW1MZ7>
"""°°°
## Load the data and the parameter for the model
°°°"""
# |%%--%%| <J9uLvW1MZ7|ZPVUGgxqlT>

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

# |%%--%%| <ZPVUGgxqlT|l145jVBiU2>

# if you are running the notebook on Google-Colab or you don't have enough RAM 
# you can reduce the number of training/test samples by using only a subset
# of all sessions. In total there are 4 sessions (1,2,3,4). Uncomment the
# lines below to use only the first 2 sessions and the first 77 seqIDs

"""
sessions_use = [1.,2.]
seq_ids_use = np.arange(1,78,1)

train_ids = np.isin(Y_train[:,Y_columns['session']], sessions_use)
test_ids  = np.isin(Y_test[:,Y_columns['session']], sessions_use) 

X_train = X_train[train_ids]
Y_train = Y_train[train_ids]
X_test = X_test[test_ids]
Y_test = Y_test[test_ids]

train_ids = np.isin(Y_train[:,Y_columns['seqId']], seq_ids_use)
test_ids  = np.isin(Y_test[:,Y_columns['seqId']], seq_ids_use) 

X_train = X_train[train_ids]
Y_train = Y_train[train_ids]
X_test = X_test[test_ids]
Y_test = Y_test[test_ids]
"""

# |%%--%%| <l145jVBiU2|c6HJmL2Sgb>

batch_size = 64

# if the batch_size is too high for your GPU/CPU uncomment the following line
#batch_size = 32

# |%%--%%| <c6HJmL2Sgb|W8JxMgBu6k>
"""°°°
### Show example instances of the data
°°°"""
# |%%--%%| <W8JxMgBu6k|2DVS45yTOt>

from IPython.display import Image
Image(filename='images/scanpaths.jpg')

# |%%--%%| <2DVS45yTOt|i7ACqZ5Zxa>
"""°°°
## Apply the Data Augmentation/Enhancement
* Here you should implement your own data augmentation/enhancement
* implement the function 'transform(X_train,Y_train,Y_columns)' in 'DataAugmentation/data_augmentation.py'
°°°"""
# |%%--%%| <i7ACqZ5Zxa|MH1tnq76X7>

from DataAugmentation import data_augmentation
X_train_augmented,Y_train_augmented = data_augmentation.transform(X_train,Y_train,Y_columns)

# |%%--%%| <MH1tnq76X7|vu4hMdrr1o>
"""°°°
## Train model and get embeddings
°°°"""
# |%%--%%| <vu4hMdrr1o|94j073SeiY>

embeddings_concatenated_augmented = evaluation.evaluate_create_test_embeddings(X_train_augmented,
                                                                               Y_train_augmented,
                                                                               X_test,Y_test,
                                                                               Y_columns,
                                                                               batch_size = batch_size)

# |%%--%%| <94j073SeiY|EtnoPpbWhA>

embeddings_concatenated_baseline = evaluation.evaluate_create_test_embeddings(X_train,
                                                                              Y_train,
                                                                              X_test,
                                                                              Y_test,
                                                                              Y_columns,
                                                                              batch_size = batch_size)

# |%%--%%| <EtnoPpbWhA|XoWI9shvNG>

num_sessions = len(np.unique(Y_test[:,Y_columns['session']]))
print('number of sessions: ' + str(num_sessions))

# |%%--%%| <XoWI9shvNG|tRimdR9mNE>
"""°°°
## Plot results
°°°"""
# |%%--%%| <tRimdR9mNE|Cp8CHgkaWk>

window_sizes = [1,5,10]
n_train_users = 0
n_enrolled_users = 1
n_impostors = 24
n_enrollment_sessions = num_sessions -1
n_test_sessions = 1
test_user = None
test_sessions = None
user_test_sessions = None 
enrollment_sessions = None
verbose = 0
random_state = 42,
seconds_per_session = None
model_names = ['DeepEye baseline',
              'DeepEye augment']
embedding_list = [embeddings_concatenated_baseline,embeddings_concatenated_augmented]
metric_lists = []
for i in range(len(model_names)):
    metric_dict = dict()
    for random_state in tqdm(np.arange(10)):
        from sklearn import metrics
        (score_dicts, label_dicts) = evaluation.get_scores_and_labels_from_raw(
                                    test_embeddings=embedding_list[i],
                                    Y_test=Y_test,
                                    Y_columns=Y_columns,
                                    window_sizes=window_sizes,
                                    n_train_users = n_train_users,
                                    n_enrolled_users = n_enrolled_users,
                                    n_impostors = n_impostors,
                                    n_enrollment_sessions = n_enrollment_sessions,
                                    n_test_sessions = n_test_sessions,
                                    test_user = test_user,
                                    test_sessions = test_sessions,
                                    user_test_sessions = user_test_sessions,
                                    enrollment_sessions = enrollment_sessions,
                                    verbose = verbose,
                                    random_state = random_state,
                                    seconds_per_session = seconds_per_session)

        for window_size in window_sizes:
            window_size = str(window_size)
            cur_scores = score_dicts[window_size]
            cur_label  = label_dicts[window_size]
            fpr, tpr, thresholds = metrics.roc_curve(cur_label, cur_scores, pos_label=1)
            if window_size not in metric_dict:
                metric_dict[window_size] = dict()
            if 'fpr' not in  metric_dict[window_size]:
                metric_dict[window_size]['fpr'] = []
                metric_dict[window_size]['tpr'] = []
            metric_dict[window_size]['fpr'].append(fpr)
            metric_dict[window_size]['tpr'].append(tpr)
    metric_lists.append(metric_dict)


for window_size in window_sizes:
    for i in range(len(model_names)):
        metric_dict = metric_lists[i]
        model_name = model_names[i]
        if i == 0:
            plot_random = True
        else:
            plot_random = False
        window_size = str(window_size)
        evaluation.avg_fnr_fpr_curve(metric_dict[window_size]['fpr'], metric_dict[window_size]['tpr'], 
                label = model_name, plot_random=plot_random,
                title = 'Window_size: ' + window_size, plot_statistics = False,
                loc = 'best', plot_legend = True,
                plot_points = 1000, ncol=1,
                bbox_to_anchor=None,
                starting_point = None,
                fontsize = 14, xscale = 'log',
                setting = 'verification')
    plt.show()

# |%%--%%| <Cp8CHgkaWk|l0GIUHERMG>

window_sizes = [1,5,10]
n_train_users = 0
n_enrolled_users = 20
n_impostors = 5
n_enrollment_sessions = num_sessions -1
n_test_sessions = 1
test_user = None
test_sessions = None
user_test_sessions = None 
enrollment_sessions = None
verbose = 0
random_state = 42,
seconds_per_session = None
model_names = ['DeepEye baseline',
              'DeepEye augment']
embedding_list = [embeddings_concatenated_baseline,embeddings_concatenated_augmented]
metric_lists = []
for i in range(len(model_names)):
    metric_dict = dict()
    for random_state in tqdm(np.arange(10)):
        from sklearn import metrics
        (score_dicts, label_dicts) = evaluation.get_scores_and_labels_from_raw(
                                    test_embeddings=embedding_list[i],
                                    Y_test=Y_test,
                                    Y_columns=Y_columns,
                                    window_sizes=window_sizes,
                                    n_train_users = n_train_users,
                                    n_enrolled_users = n_enrolled_users,
                                    n_impostors = n_impostors,
                                    n_enrollment_sessions = n_enrollment_sessions,
                                    n_test_sessions = n_test_sessions,
                                    test_user = test_user,
                                    test_sessions = test_sessions,
                                    user_test_sessions = user_test_sessions,
                                    enrollment_sessions = enrollment_sessions,
                                    verbose = verbose,
                                    random_state = random_state,
                                    seconds_per_session = seconds_per_session)

        for window_size in window_sizes:
            window_size = str(window_size)
            cur_scores = score_dicts[window_size]
            cur_label  = label_dicts[window_size]
            fpr, tpr, thresholds = metrics.roc_curve(cur_label, cur_scores, pos_label=1)
            if window_size not in metric_dict:
                metric_dict[window_size] = dict()
            if 'fpr' not in  metric_dict[window_size]:
                metric_dict[window_size]['fpr'] = []
                metric_dict[window_size]['tpr'] = []
            metric_dict[window_size]['fpr'].append(fpr)
            metric_dict[window_size]['tpr'].append(tpr)
    metric_lists.append(metric_dict)


for window_size in window_sizes:
    for i in range(len(model_names)):
        metric_dict = metric_lists[i]
        model_name = model_names[i]
        if i == 0:
            plot_random = True
        else:
            plot_random = False
        window_size = str(window_size)
        evaluation.avg_fnr_fpr_curve(metric_dict[window_size]['fpr'], metric_dict[window_size]['tpr'], 
                label = model_name, plot_random=plot_random,
                title = 'Window_size: ' + window_size, plot_statistics = False,
                loc = 'best', plot_legend = True,
                plot_points = 1000, ncol=1,
                bbox_to_anchor=None,
                starting_point = None,
                fontsize = 14, xscale = 'log',
                setting = 'identification')
    plt.show()

# |%%--%%| <l0GIUHERMG|Xro1yzUigA>


