from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline


def transform(X_train, Y_train, Y_columns):
    sub_ids = Y_train[:, Y_columns['subId']]
    unique_sub_ids = list(np.unique(sub_ids))
    use_sub_ids = unique_sub_ids[0:25]
    use_ids = np.isin(sub_ids, use_sub_ids)
    X_train = X_train[use_ids]
    Y_train = Y_train[use_ids]
    return X_train, Y_train


def jitter(sample, std):
    to_jitter = np.random.normal(0, std, (1000,4))
    return sample + to_jitter


def mag_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[0])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1], knot+2))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0]-1., num=knot+2))).T
    warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[dim,:])(orig_steps) for dim in range(x.shape[1])]).T
    return x * warper

def generate_warper(x, sigma, knot):
    orig_steps = np.arange(x.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[1], knot+2))
    warp_steps = (np.ones((x.shape[1], 1)) * (np.linspace(0, x.shape[0]-1., num=knot+2))).T
    warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[dim,:])(orig_steps) for dim in range(x.shape[1])]).T
    return warper


def convert_to_xy(sample_train):
    x = [0]
    y = [0]
    for x_new, y_new in zip(sample_train[:, 0], sample_train[:, 1]):
        x.append(x[-1] + x_new)
        y.append(y[-1] + y_new)
    return np.vstack([np.array(x), np.array(y)])


def center(x):
    return np.vstack([x[0,:] - np.mean(x[0,:]), x[1,:] - np.mean(x[1,:])])

def convert_to_angles(sample_train):
    x = []
    y = []
    for idx, (x_new, y_new) in enumerate(zip(sample_train[0, 1:], sample_train[1, 1:])):
        x.append(x_new - sample_train[0, idx])
        y.append(y_new - sample_train[1, idx])
    return np.vstack([x, y])

def rotate(sample_train, rotation_angle):
    rotation_mat = np.array([[np.cos(rotation_angle), - np.sin(rotation_angle)],
                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
    return rotation_mat @ sample_train


def rotate_2d(x, max_rotation):
    left_eye = x[:, 0:2]
    right_eye = x[:, 2:4]
    
    left_eye = convert_to_xy(left_eye)
    right_eye = convert_to_xy(right_eye)

    left_eye = center(left_eye) 
    right_eye = center(right_eye) 

    rotation_angle = np.random.uniform(-max_rotation, max_rotation)
    left_eye = rotate(left_eye, rotation_angle)
    right_eye = rotate(right_eye, rotation_angle)

    left_eye = convert_to_angles(left_eye)
    right_eye = convert_to_angles(right_eye)

    return np.hstack([left_eye.T, right_eye.T])


def filter_high_std(X_train, Y_train, cutoff=0.065):
    stds_per_row_all = np.mean(np.std(X_train, axis=1), axis=1)
    filter_mask = stds_per_row_all < cutoff
    return X_train[filter_mask,:,:], Y_train[filter_mask, :]


def filter_low_std(X_train, Y_train, cutoff=0.025):
    stds_per_row_all = np.mean(np.std(X_train, axis=1), axis=1)
    filter_mask = stds_per_row_all > cutoff
    return X_train[filter_mask,:,:], Y_train[filter_mask, :]


def augment_all_in_place(X_train, Y_train, aug_config, original_length=108000):
    if "no_aug" in aug_config:
        return 

    num_of_repititions = aug_config['perc_augmentation']

    num_of_iterations = [(x, range_run) for range_run in range(num_of_repititions) for x in range(original_length)]
    num_of_runs = ( (x, range_run, X_train[x].copy(), aug_config) for x, range_run in num_of_iterations )
    with Pool(processes=12) as p:
        for x, range_run, to_aug in tqdm(p.imap_unordered(augment_processor, num_of_runs), total=original_length * num_of_repititions):
        # for range_run in tqdm(range(num_of_repititions)):
        #     for x in tqdm(range(original_length), leave=False):
                # to_aug = X_train[x].copy()
            index_to_replace = ((range_run + 1) * original_length) + x
            # to_aug = augmentation_chooser(to_aug, aug_config)
            X_train[index_to_replace] = to_aug
            Y_train[index_to_replace] = Y_train[x]


def augment_all(X_train, Y_train, aug_config):
    if "no_aug" in aug_config:
        return X_train, Y_train

    X_aug, Y_aug = [], []
    num_of_repititions = aug_config['perc_augmentation']
    for _ in tqdm(range(num_of_repititions)):
        for x in tqdm(range(len(X_train)), leave=False):
            to_aug = X_train[x].copy()
            for aug_type, hyper_params in aug_config.items():
                if aug_type == 'jitter':
                    to_aug = jitter(to_aug, hyper_params)
                if aug_type == 'mag_warp':
                    if type(hyper_params) == tuple or type(hyper_params) == list:
                        to_aug = mag_warp(to_aug, hyper_params[0], hyper_params[1])
                    else:
                        to_aug = mag_warp(to_aug, hyper_params)
                if aug_type == 'rotate':
                    to_aug = rotate_2d(to_aug, hyper_params)
            X_aug.append(to_aug)
            Y_aug.append(Y_train[x])

    for x in range(len(X_train)):
        X_aug.append(X_train[x])
        Y_aug.append(Y_train[x])
            
    return np.array(X_aug), np.array(Y_aug)


def augmentation_chooser(to_aug, aug_config):
    for aug_type, hyper_params in aug_config.items():
        if aug_type == 'jitter':
            to_aug = jitter(to_aug, hyper_params)
        if aug_type == 'mag_warp':
            if type(hyper_params) == tuple or type(hyper_params) == list:
                to_aug = mag_warp(to_aug, hyper_params[0], hyper_params[1])
            else:
                to_aug = mag_warp(to_aug, hyper_params)
        if aug_type == 'rotate':
            to_aug = rotate_2d(to_aug, hyper_params)
    return to_aug


def augment_processor(param):
    x, range_run, to_aug, aug_config = param
    to_aug = augmentation_chooser(to_aug, aug_config)
    return x, range_run, to_aug
