import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import seed
from scipy import interpolate
from scipy.spatial import distance
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from Model import configuration
from Model import deepeyedentification_tf2

model_conf = {"learning_rate_merged": 0.00011, "kernel_sub2": [9, 9, 9, 5, 5, 5, 5, 3, 3], "normalization_sub2": ["zscore"], "kernel_sub1": [9, 9, 9, 5, 5, 5, 5, 3, 3], "transform_sub2": ["clip", 0.01], "filters_sub1": [128, 128, 128, 256, 256, 256, 256, 256, 256], "dense_sub1": [256, 256, 128], "learning_rate_sub2": 0.001, "normalization_sub1": 'None', "name_sub1": "optimal_slow_subnet", "learning_rate_sub1": 0.001, "filters_sub2": [32, 32, 32, 512, 512, 512, 512, 512, 512], "name_merged": "optimal_merged", "Ndense_merged": [256, 128], "dense_sub2": [256, 256, 128], "strides_sub1": [1, 1, 1, 1, 1, 1, 1, 1, 1], "transform_sub1": ["tanh", 20.0], "name_sub2": "optimal_fast_subnet", "strides_sub2": [1, 1, 1, 1, 1, 1, 1, 1, 1],
"batch_size":64}


def avg_fnr_fpr_curve(
    fprs, tprs, label, plot_random=False,
    title=None, plot_statistics=False,
    loc='best', plot_legend=True,
    plot_points=10000, ncol=1,
    bbox_to_anchor=None,
    starting_point=None,
    fontsize=14, xscale=None,
    setting='verification',
):
    """
    Plot average roc curve from multiple fpr and tpr arrays of multiple cv-folds

    :param fprs: list of fpr arrays for the different folds
    :param tprs: list of tpr arrays for the different folds
    :label: name for the legend
    :plot_random: indicator, indicating if the random guessing curve should be plotted
    :title: title of plot; no title if 'None'
    :plot_statistics: if True, statistics for all the folds are plotted
    :loc: location of legend
    :plot_legend: if True legend is plotted
    :plot_points: number of points to plot
    :ncol: number of columns for legend
    :bbox_to_anchor: bounding box for legend outside of plot
    :starting_point: indicates the starting point of drawing the curves
    :fontsize: fontsize
    :xscale: scale for x-axis
    :setting: verification or identification
    """
    if xscale is not None:
        plt.xscale(xscale)

    tprs_list = []
    aucs = []
    audets = [] 
    for i in range(0, len(fprs)):
        fpr = fprs[i]
        fnr = 1 - tprs[i]

        tprs_list.append(interpolate.interp1d(fpr, fnr))
        aucs.append(metrics.auc(fprs[i], tprs[i]))
        audets.append(metrics.auc(fpr, fnr))
    aucs = np.array(aucs)
    audets = np.array(audets)
    x = np.linspace(0, 1, plot_points)
    if starting_point is not None:
        x = x[x > starting_point]

    if plot_random:
        y = 1 - x
        plt.plot(
            x, y, color='grey', linestyle='dashed',
            label='random guessing',
        )

    # plot average and std error of those roc curves:
    ys = np.vstack([f(x) for f in tprs_list])
    ys_mean = ys.mean(axis=0)
    ys_std = ys.std(axis=0) / np.sqrt(len(fprs))

    distances = np.abs(ys_mean - x)
    min_distance_idx = np.argmin(distances)
    cur_label = str(label)
    if plot_statistics:
        # cur_label += r' (AUC={} $\pm$ {})'.format(
        #     np.round(np.mean(aucs), 4),
        #     np.round(np.std(aucs), 4),
        # )
        cur_label += f' EER={ys_mean[min_distance_idx]} '
        cur_label += r' (AUDET={} $\pm$ {})'.format(
            np.round(np.mean(audets), 4),
            np.round(np.std(audets), 4),
        )
    plt.plot(x, ys_mean, label=cur_label)
    plt.fill_between(x, ys_mean - ys_std, ys_mean + ys_std, alpha=0.2)
    if plot_legend:
        if bbox_to_anchor is None:
            plt.legend(loc=loc, ncol=ncol, fontsize=fontsize)
        else:
            plt.legend(
                loc=loc, ncol=ncol,
                bbox_to_anchor=bbox_to_anchor, fontsize=fontsize,
            )

    if setting == 'verification':
        plt.xlabel('FMR', fontsize=fontsize)
        plt.ylabel('FNMR', fontsize=fontsize)
    elif setting == 'identification':
        plt.xlabel('FPIR', fontsize=fontsize)
        plt.ylabel('FNIR', fontsize=fontsize)
    # plt.ticklabel_format(useOffset=False, style='plain')

    plt.grid('on')
    if title is not None:
        plt.title(title)

    return aucs


def get_indicies(
    enrolled_users,
    impostors,
    enrollment_sessions,
    test_sessions,
    data_user,
    data_sessions,
    data_seqIds,
    seconds_per_session=None,
    random_state=42,
):

    idx_enrollment = np.zeros((data_user.shape[0], 3), dtype=bool)
    if data_seqIds is None:
        num_enrollment = 12
        idx_enrollment[:, 0] = np.logical_and(
            np.isin(data_user, enrolled_users),
            np.isin(data_sessions, enrollment_sessions[0:1]),
        )
        pos_ids = np.where(idx_enrollment[:, 0])[0]
        random.shuffle(pos_ids)
        use_ids = pos_ids[0:num_enrollment]
        idx_enrollment = np.zeros((data_user.shape[0], 3), dtype=bool)
        idx_enrollment[use_ids, 0] = True
        idx_enrollment[use_ids, 1] = True
        idx_enrollment[use_ids, 2] = True
    else:
        if seconds_per_session is not None:
            random.seed(random_state)
            # select random seconds from enrollment
            for en_i in range(len(enrollment_sessions)):
                for us_i in range(len(enrolled_users)):
                    cur_user_ids = np.where(
                        np.logical_and(
                            np.isin(data_user, enrolled_users[us_i]),
                            np.isin(
                                data_sessions,
                                enrollment_sessions[en_i:en_i+1],
                            ),
                        ),
                    )[0]
                    random.shuffle(cur_user_ids)
                    cur_user_use_ids = cur_user_ids[0:seconds_per_session]
                    idx_enrollment[cur_user_use_ids, :] = True
        else:
            # 1 enrollment session: 12 trials, 2 enrollment sessions: 6+6 trials, 3 enrollment sessions: 4+4+4 trials
            seqIds_different_configs = [
                109, 61, 13,
                97, 49, 3, 133, 85, 37, 121, 73, 25,
            ]

            # one enrollment session with 12 unique trial configs:
            idx_enrollment[:, 0] = np.logical_and(
                np.isin(data_user, enrolled_users),
                np.logical_and(
                    np.isin(data_sessions, enrollment_sessions[0:1]),
                    np.isin(data_seqIds, seqIds_different_configs),
                ),
            )
            # two enrollment session with 12 unique trial configs:
            idx_enrollment[:, 1] = np.logical_and(
                np.isin(data_user, enrolled_users),
                np.logical_or(
                    np.logical_and(
                        np.isin(data_sessions, enrollment_sessions[0:1]),
                        np.isin(
                            data_seqIds,
                            seqIds_different_configs[0:6],
                        ),
                    ),
                    np.logical_and(
                        np.isin(data_sessions, enrollment_sessions[1:2]),
                        np.isin(
                            data_seqIds,
                            seqIds_different_configs[6:12],
                        ),
                    ),
                ),
            )
            # three enrollment session with 12 unique trial configs:
            idx_enrollment[:, 2] = np.logical_and(
                np.isin(data_user, enrolled_users),
                np.logical_or(
                    np.logical_or(
                        np.logical_and(
                            np.isin(
                                data_sessions,
                                enrollment_sessions[0:1],
                            ),
                            np.isin(
                                data_seqIds,
                                seqIds_different_configs[0:4],
                            ),
                        ),
                        np.logical_and(
                            np.isin(
                                data_sessions,
                                enrollment_sessions[1:2],
                            ),
                            np.isin(
                                data_seqIds,
                                seqIds_different_configs[4:8],
                            ),
                        ),
                    ),
                    np.logical_and(
                        np.isin(data_sessions, enrollment_sessions[2:3]),
                        np.isin(
                            data_seqIds,
                            seqIds_different_configs[8:12],
                        ),
                    ),
                ),
            )

    test_idx = np.logical_and(
        np.logical_or(
            np.isin(data_user, enrolled_users),
            np.isin(data_user, impostors),
        ),
        np.isin(data_sessions, test_sessions),
    )

    return (idx_enrollment, test_idx)


def get_user_similarity_scores_and_labels(
    cosine_distances, y_enrollment, y_test, enrollment_users, impostors, window_size=1,
    sim_to_enroll='min',
    verbose=0,
):
    """

    :param cosine_distances: cosine distances of all pairs of enrollment and test instances, n_test x n_enrollment
    :param y_enrollment: n_enrollment labels for enrollment instances
    :param y_test: n_test labels for test instances
    :param enrollment_users: all ids of enrolled users
    :param impostors: all ids of impostors
    :param window_size: number of instances the similarity score should be based upon
    :param sim_to_enroll: how to compute simalarity to enrollment users; should be in {'min','mean'}
    :return: similarity scores of two persons; true labels: test person is impostor (0), same person (1) or another enrolled person (2)
    """
    if verbose == 0:
        disable = True
    else:
        disable = False

    scores = []      # similarity score between two users, based on number of test instances specified by window size
    # true labels: test person is 0 (impostor),1 (correct), 2 (confused)
    labels = []

    for test_user in tqdm(np.unique(y_test), disable=disable):
        idx_test_user = y_test == test_user

        # iterate over each possible window start position for test user
        dists_test_user = cosine_distances[idx_test_user, :]
        if str(window_size) != 'all':
            for i in range(dists_test_user.shape[0] - window_size):
                dists_test_user_window = dists_test_user[i:i+window_size, :]

                # calculate score and prediction and create true label for each window
                distances_to_enrolled = []
                enrolled_u = []
                enrolled_persons = np.unique(y_enrollment)

                for enrolled_user in enrolled_persons:
                    idx_enrolled_user = y_enrollment == enrolled_user

                    # calculate aggregated distance of instances in window with each enrolled user seperately
                    dists_test_user_window_enrolled_user = dists_test_user_window[
                        :,
                        idx_enrolled_user
                    ]

                    # aggregate distances for each test sequence to all enrolled sequences by taking the minimum distance
                    if sim_to_enroll == 'min':
                        dists_test_sequences_of_window = np.min(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array
                    elif sim_to_enroll == 'mean':
                        dists_test_sequences_of_window = np.mean(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array

                    # aggregate min distances of all test sequences in this window by taking the mean
                    window_mean_dist = np.mean(dists_test_sequences_of_window)

                    distances_to_enrolled.append(window_mean_dist)
                    enrolled_u.append(enrolled_user)

                    # create corresponding true label for this window
                    if test_user in list(impostors):
                        label = 0  # test user of this window is an impostor
                    elif test_user in list(enrollment_users):
                        if test_user == enrolled_user:
                            label = 1  # test user of this window is this enrolled user
                        else:
                            label = 2  # test user of this window is another enrolled user
                    else:
                        print(
                            f'user {test_user} is neither enrolled user nor impostor',
                        )
                        label = -1  # should never happen

                    scores.append(1-window_mean_dist)
                    labels.append(label)
        else:
            dists_test_user_window = dists_test_user

            # calculate score and prediction and create true label for each window
            distances_to_enrolled = []
            enrolled_u = []
            enrolled_persons = np.unique(y_enrollment)

            for enrolled_user in enrolled_persons:
                idx_enrolled_user = y_enrollment == enrolled_user

                # calculate aggregated distance of instances in window with each enrolled user seperately
                dists_test_user_window_enrolled_user = dists_test_user_window[
                    :,
                    idx_enrolled_user
                ]

                # aggregate distances for each test sequence to all enrolled sequences by taking the minimum distance
                if sim_to_enroll == 'min':
                    dists_test_sequences_of_window = np.min(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array
                elif sim_to_enroll == 'mean':
                    dists_test_sequences_of_window = np.mean(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array

                # aggregate min distances of all test sequences in this window by taking the mean
                window_mean_dist = np.mean(dists_test_sequences_of_window)

                distances_to_enrolled.append(window_mean_dist)
                enrolled_u.append(enrolled_user)

                # create corresponding true label for this window
                if test_user in list(impostors):
                    label = 0  # test user of this window is an impostor
                elif test_user in list(enrollment_users):
                    if test_user == enrolled_user:
                        label = 1  # test user of this window is this enrolled user
                    else:
                        label = 2  # test user of this window is another enrolled user
                else:
                    print(
                        f'user {test_user} is neither enrolled user nor impostor',
                    )
                    label = -1  # should never happen

                scores.append(1-window_mean_dist)
                labels.append(label)

    return np.array(scores), np.array(labels)


def get_scores_and_labels_from_raw(
    test_embeddings, Y_test, Y_columns,
    window_sizes,
    n_train_users=0,
    n_enrolled_users=20,
    n_impostors=5,
    n_enrollment_sessions=3,
    n_test_sessions=1,
    test_user=None,
    user_test_sessions=None,
    enrollment_sessions=None,
    test_sessions=None,
    verbose=1,
    random_state=None,
    seconds_per_session=None,
):

    if random_state is not None:
        random.seed(random_state)

    score_dicts = dict()
    label_dicts = dict()

    if Y_test is None:
        test_seqIds = None
    else:
        test_user = Y_test[:, Y_columns['subId']]
        test_sessions = Y_test[:, Y_columns['session']]
        try:
            test_seqIds = Y_test[:, Y_columns['seqId']]
        except:
            test_seqIds = None
    # print('number of different users: ' + str(len(np.unique(test_user))))

    users = list(np.unique(test_user))
    if Y_test is None:
        idx_test_session = np.isin(user_test_sessions, test_sessions)
        users = list(np.unique(test_user[idx_test_session]))

    # shuffle users
    random.shuffle(users)

    enrolled_users = users[n_train_users:n_train_users+n_enrolled_users]
    impostors = users[
        n_train_users +
        n_enrolled_users: n_train_users + n_enrolled_users + n_impostors
    ]

    # JuDo setting
    if Y_test is not None:
        sessions = np.unique(test_sessions)
        random.shuffle(sessions)
        cur_enrollment_sessions = sessions[0: n_enrollment_sessions]
        cur_test_sessions = sessions[
            n_enrollment_sessions:
            n_enrollment_sessions + n_test_sessions
        ]
    else:
        random.shuffle(enrollment_sessions)
        random.shuffle(test_sessions)
        cur_enrollment_sessions = enrollment_sessions[0: n_enrollment_sessions]
        cur_test_sessions = test_sessions[0:n_test_sessions]

    if verbose > 0:
        print(
            f'enrolled_users: {enrolled_users} enroll-sessions: {cur_enrollment_sessions} test-sessions: {cur_test_sessions}',
        )

    (idx_enrollment, test_idx) = get_indicies(
        enrolled_users,
        impostors,
        cur_enrollment_sessions,
        cur_test_sessions,
        test_user,
        test_sessions,
        test_seqIds,
        seconds_per_session=seconds_per_session,
        random_state=random_state,
    )


    test_feature_vectors = test_embeddings[test_idx, :]
    enrollment_feature_vectors = test_embeddings[
        idx_enrollment[
            :,
            n_enrollment_sessions-1
        ], :
    ]

    # labels for embedding feature vectors:
    y_enrollment_user = test_user[idx_enrollment[:, n_enrollment_sessions-1]]
    y_test_user = test_user[test_idx]


    dists = distance.cdist(
        test_feature_vectors,
        enrollment_feature_vectors, metric='cosine',
    )

    for window_size in window_sizes:
        scores, labels = get_user_similarity_scores_and_labels(
            dists,
            y_enrollment_user,
            y_test_user,
            enrolled_users,
            impostors,
            window_size=window_size,
            verbose=verbose,
        )
        cur_key = str(window_size)
        score_dicts[cur_key] = scores.tolist()
        label_dicts[cur_key] = labels.tolist()
    return (score_dicts, label_dicts)


def evaluate_create_test_embeddings(
    X_train,
    Y_train,
    X_test,
    Y_test,
    Y_columns,
    batch_size = 64,
    augmentation_type=None):
    
    # clear tensorflow session
    tf.keras.backend.clear_session()

    # Set Seeds
    seed(42)
    # select graphic card
    tf_config = tf.compat.v1.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf_config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=tf_config)
    tf.random.set_seed(42)
    
    if batch_size != -1:
        model_conf['batch_size'] = batch_size
    
    # load  model configuration
    conf = configuration.load_config(model_conf)

    
    # encode label
    le = LabelEncoder()
    le.fit(Y_train[:, Y_columns['subId']])
    Y_train[:, Y_columns['subId']] = le.transform(
        Y_train[:, Y_columns['subId']],
    )

    # one-hot-encode user ids:
    n_train_users_f = len(np.unique(Y_train[:, Y_columns['subId']]))
    y_train = to_categorical(
        Y_train[:, Y_columns['subId']], num_classes=n_train_users_f,
    )

    # SET UP PARAMS FOR NN

    # calculate z-score normalization for fast subnet and add it to configuration:
    m = np.mean(X_train[:, :, [0, 1]], axis=None)
    sd = np.std(X_train[:, :, [0, 1]], axis=None)

    conf.subnets[1].normalization = conf.subnets[1].normalization._replace(
        mean=m, std=sd,
    )

    seq_len = X_train.shape[1]
    n_channels = X_train.shape[2]
    n_classes = y_train.shape[1]

    X_diffs_train = X_train[:, :, [0, 1]] - X_train[:, :, [2, 3]]
    mean_vel_diff = np.nanmean(X_diffs_train)
    std_vel_diff = np.nanstd(X_diffs_train)

    X_diffs_test = X_test[:, :, [0, 1]] - X_test[:, :, [2, 3]]

    # TRAIN NN
    deepeye = deepeyedentification_tf2.DeepEyedentification2Diffs(
        conf.subnets[0],
        conf.subnets[1],
        conf,
        seq_len=seq_len,
        channels=n_channels,
        n_classes=n_classes,
        zscore_mean_vel_diffs=mean_vel_diff,
        zscore_std_vel_diffs=std_vel_diff,
    )

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train, Y_train[:, Y_columns['subId']]):
        break

    print('Training model on train data ...')

    # train the model
    cur_hist = deepeye.train(
        X_train, X_diffs_train, y_train,
        train_idx=train_index, validation_idx=val_index,
        augmentation_type=augmentation_type,
        # pretrained_weights_fast_path="trained_models/",
        # pretrained_weights_slow_path="trained_models/"
    )
    print('done.')

    from tensorflow.keras import Model
    print('Creating embedding for test data ...')
    embedding_fast_subnet = Model(
        inputs=deepeye.fast_subnet.input,
        outputs=deepeye.fast_subnet.get_layer('fast_d3').output,
    )
    embedding_slow_subnet = Model(
        inputs=deepeye.slow_subnet.input,
        outputs=deepeye.slow_subnet.get_layer('slow_d3').output,
    )
    embedding_deepeye = Model(
        inputs=deepeye.model.input,
        outputs=deepeye.model.get_layer('deepeye_a2').output,
    )

    embeddings_fast_subnet_all = embedding_fast_subnet.predict(
        [X_test, X_diffs_test],
        batch_size=8
    )
    embeddings_slow_subnet_all = embedding_slow_subnet.predict(
        [X_test, X_diffs_test],
        batch_size=8
    )
    embeddings_deepeye_all = embedding_deepeye.predict(
        [[X_test, X_diffs_test], [X_test, X_diffs_test]],
        batch_size=8
    )

    embeddings_concatenated_all = np.hstack([
        embeddings_fast_subnet_all,
        embeddings_slow_subnet_all,
        embeddings_deepeye_all,
    ])
    print('done.')
    return embeddings_concatenated_all
