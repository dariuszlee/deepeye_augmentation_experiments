import pipes
from subprocess import Popen
import subprocess
import json
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm
import pickle
import numpy as np
np.random.seed(42)
import boto3
import tensorflow as tf
tf.random.set_seed(42)
import os
import socket
from Evaluation import evaluation
from tqdm import tqdm

print(socket.gethostname())

import os

instance_names = 'eks-MLScaler'

# session = boto3.Session(region_name='eu-central-1')
# sqs = boto3.client('sqs')
# s3 = session.resource('s3')

os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'

copy_resources_all = "scp -rp -i dariusec2.pem ./evaluate.py ./requirements.txt ./configs ./Data ./DataAugmentation ./deepEye_data_augmentation_research_module.ipynb ./deepEye_data_augmentation_research_module.py ./deepEye_data_augmentation_research_module_rm_evaluation.ipynb ./deepeye_experiment_visualization.ipynb ./deepeye_experiment_visualization.py ./env.list ./evaluate_multiple.py ./Evaluation ./images ./magnitude_data_augmentation_experiments.ipynb ./magnitude_data_augmentation_experiments.py ./Model ./README.md ./trained_models ec2-user@{SERVER}:/home/ec2-user/"


copy_code = "scp -rp -i dariusec2.pem ./evaluate.py ./DataAugmentation ./deepEye_data_augmentation_research_module.py ./deepeye_experiment_visualization.py ./evaluate_multiple.py ./Evaluation ./magnitude_data_augmentation_experiments.py ./Model ./requirements.txt ec2-user@{SERVER}:/home/ec2-user/"

commands_start_up = [
    "sudo amazon-linux-extras install python3.8",
    "pip3.8 install -r requirements.txt"
]
instance_type = "p2.xlarge" ## "g3s.xlarge" was too small
aws_ami = 'Deep Learning AMI GPU TensorFlow 2.10.0 (Ubuntu 20.04) 20221104'
servers = [
    'ec2-3-121-78-202.eu-central-1.compute.amazonaws.com',
    'ec2-52-58-49-19.eu-central-1.compute.amazonaws.com',
    'ec2-18-184-175-55.eu-central-1.compute.amazonaws.com',
    'ec2-18-198-22-150.eu-central-1.compute.amazonaws.com',
    'ec2-18-184-207-94.eu-central-1.compute.amazonaws.com'
]

# QUEUE_URL = os.environ["WORKER_QUEUE_URL"]
# BUCKET_NAME = os.environ["BUCKET_NAME"]
QUEUE_URL = "https://sqs.eu-central-1.amazonaws.com/624058868395/MLWorkerQueue"
BUCKET_NAME = "kontist-ml-models"

def push_to_s3(data, file_name):
    metric_data = pickle.dumps(data)

    object = s3.Object(BUCKET_NAME, json.dumps(file_name))

    result = object.put(Body=metric_data)
    print(result)


def send_to_queue(config):
    # Send message to SQS queue
    response = sqs.send_message(
        QueueUrl=QUEUE_URL,
        DelaySeconds=10,
        MessageAttributes={
            'Type': {
                'DataType': 'String',
                'StringValue': 'predict'
            },
        },
        MessageBody=(json.dumps(config))
    )
    return response


def read_from_queue():
    response = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        MessageAttributeNames=[
            'All'
        ],
        VisibilityTimeout=0,
        WaitTimeSeconds=20
    )

    if 'Messages' in response and 'ReceiptHandle' in response['Messages'][0]:
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
    else:
        message = None
        receipt_handle = None

    data = json.loads(message['Body'])

    sqs.delete_message(
        QueueUrl=QUEUE_URL,
        ReceiptHandle=receipt_handle
    )
    return data


def initialize_tests():
    solo_configurations = []
    # for percentage_augmentation in [0.3, 0.6, 1]:
    # for percentage_augmentation in [0.3]:
    for run in range(0, 3):
    # for percentage_augmentation in [1]:
        # for run in range(0, 5):
            # jitters
        solo_configurations.append({ "no_aug": {}, "perc_augmentation": 0, "run": run })
        for percentage_augmentation in [3, 1]:

            for i in [0.0001, 0.001, 0.005]:
                config = { 'jitter': np.around(i, 4), "perc_augmentation": percentage_augmentation, "run": run}
                solo_configurations.append(config)

            std = 0.15
            for knot in [2, 4, 8]:
                config = {'mag_warp': [np.around(std, 4), knot], "perc_augmentation": percentage_augmentation, "run": run}
                solo_configurations.append(config)

            # mag warp
            knot = 4
            # for i in [0.1, 0.3, 0.5, 1.0]:
            # for i in [0.5]:
            for i in [0.1, 0.3, 1.0, 1.5, 2.0, 3.0]:
                config = {'mag_warp': [np.around(i, 4), knot], "perc_augmentation": percentage_augmentation, "run": run}
                solo_configurations.append(config)

            # mag warp
            knot = 4
            for i in [0.001, 0.005, 0.01, 0.03]:
                config = {'mag_warp': [np.around(i, 4), knot], "perc_augmentation": percentage_augmentation, "run": run}
                solo_configurations.append(config)

            #rotate
            for i in [0.5, 1, 5, 180]:
                solo_configurations.append({"rotate": i, "perc_augmentation": percentage_augmentation, "run": run})

    filtered_configs = []
    for test in solo_configurations:
        key = tuple(test.items())
        if not os.path.exists(f'new_results/{key}.pkl'):
            filtered_configs.append(test)
        else:
            print(f"Skipping {key}")
    # for config in solo_configurations:
    #     send_to_queue(config) 

    print(float(len(filtered_configs)) / len(solo_configurations))
    __import__('ipdb').set_trace()
    return filtered_configs


def train_evaluate(aug_param):
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

    key = tuple(aug_param.items())
    print(f"Processing key: {key}")

    batch_size = aug_param.get('batch_size', 64)

    tf.random.set_seed(42)
    # X_train_augmented, Y_train_augmented = augment(X_train, Y_train, Y_columns, aug_param)
    X_train_augmented, Y_train_augmented = X_train, Y_train
    embeddings_concatenated_augmented = evaluation.evaluate_create_test_embeddings(X_train_augmented,
                                                                               Y_train_augmented,
                                                                               X_test,Y_test,
                                                                               Y_columns,
                                                                               batch_size = batch_size,
                                                                               augmentation_type=aug_param)

    num_sessions = len(np.unique(Y_test[:,Y_columns['session']]))
    print('number of sessions: ' + str(num_sessions))

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

    metric_dict = dict()
    for random_state in tqdm(np.arange(10)):
        from sklearn import metrics
        (score_dicts, label_dicts) = evaluation.get_scores_and_labels_from_raw(
                                    test_embeddings=embeddings_concatenated_augmented,
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

    # push_to_s3(metric_dict, key)
    experiment = {
        "config": aug_param,
        "metrics": metric_dict
    }
    if not os.path.exists("new_results"):
        os.mkdir("new_results/")

    with open(f"new_results/{key}.pkl", "wb") as f:
        pickle.dump(experiment, f)


def string_to_tuple(myStr):
    print("The tuple string is:", myStr)
    myStr = myStr.replace("(", "")
    myStr = myStr.replace(")", "")
    myStr = myStr.replace(",", " ")
    myList = myStr.split()
    myList = list(map(int, myList))
    myTuple = tuple(myList)
    return myTuple


def fetch_results(urls):
    procs = []
    for server in urls:
        # subprocess.run(["scp", "-i", "~/.ssh/dariusec2.pem", f"ec2-user@{server}:/home/ec2-user/results/*.pkl", "./new_results"])
        procs.append(Popen(["scp", "-i", "~/.ssh/dariusec2.pem", f"ec2-user@{server}:/home/ec2-user/results/*.pkl", "./new_results"]))
    for p in procs:
        p.wait()


def push_results(urls):
    procs = []
    for server in urls:
        # subprocess.run(["scp", "-i", "~/.ssh/dariusec2.pem", f"ec2-user@{server}:/home/ec2-user/results/*.pkl", "./new_results"])
        procs.append(Popen(["scp", "-r", "-i", "~/.ssh/dariusec2.pem", "./new_results/", f"ec2-user@{server}:/home/ec2-user/results/", ]))
    for p in procs:
        p.wait()


def load_results(directory='./new_results/', out='metric_dicts_new.pkl'):
    result_files = os.listdir(directory)
    all_data = {}
    for file in result_files:
        experiment_name = file.split('.pkl')[0]
        file_path = f"{directory}/{file}"
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if 'config' in data:
                if 'no_aug' in data['config']:
                    data['config']['no_aug'] = tuple()
                if "mag_warp" in data['config']:
                    data['config']['mag_warp'] = tuple(data['config']['mag_warp'])
                try:
                    all_data[tuple(data['config'].items())] = data['metrics']
                except Exception as e:
                    __import__('ipdb').set_trace()
                    print("nohting")
            else:  # old way
                all_data[experiment_name] = data
    with open(out, 'wb') as f:
        pickle.dump(all_data, f)


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def create_ec2s():
    ec2 = boto3.resource('ec2')
    instances = ec2.create_instances(
            ImageId="ami-0413089e29cebeee5",
            MinCount=5,
            MaxCount=5,
            InstanceType="g4dn.4xlarge",
            # InstanceType="g3s.xlarge",
            KeyName="dariusec2",
            TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": instance_names}]
            }
            ],
            SecurityGroups=['allow_ssh']
        )
    return instances


def get_ec2_instances():
    client = boto3.client('ec2')
    custom_filter = [{
        'Name':'tag:Name', 
        'Values': [instance_names]}
                     ]
        
    response = client.describe_instances(Filters=custom_filter)
    if len(response['Reservations']) == 0 or len(response['Reservations'][0]['Instances']) != 5:
        __import__('ipdb').set_trace()
        instances = create_ec2s()
        for instance in instances:
            instance.wait_until_running()
    else:
        return [r['PublicDnsName'] for r in response['Reservations'][0]['Instances']]


def upload_data(urls):
    procs = []
    for url in urls:
        ssh_url = f"ec2-user@{url}"
        exists = exists_remote(ssh_url, "/home/ec2-user/Data/train_data.npz")
        if not exists:
            sub_string = copy_resources_all.format(SERVER=url).split(' ')
        else:
            sub_string = copy_code.format(SERVER=url).split(' ')
        procs.append(Popen(sub_string))
    for p in procs:
        p.wait()


def exists_remote(host, path):
    status = subprocess.call(
        ['ssh', "-i", "dariusec2.pem", host, 'test -f {}'.format(pipes.quote(path))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')


def split_processes(urls, send=False):
    tests = initialize_tests()
    splits = np.array_split(np.array(tests), len(urls))
    for url, split in zip(urls, splits):
        with open("tests.json", "w") as f:
            json.dump(list(split), f)
        if send:
            subprocess.run("scp -i dariusec2.pem tests.json ec2-user@{SERVER}:/home/ec2-user/".format(SERVER=url).split(' '))
        # subprocess.Popen("ssh -i dariusec2.pem ec2-user@{SERVER} screen -d -m python3.9 evaluate.py".format(SERVER=url).split(' '))
        print(url)


def main(tests):
    # fix_gpu()
    for test in tests:
        key = tuple(test.items())
        if not os.path.exists(f'new_results/{key}.pkl'):
            train_evaluate(test)
        else:
            print(f"Skipping {key}")
    fetch_results()
    # load_results()


if __name__ == "__main__":
    load_results('./new_results/', 'metric_dict.pkl')
    # split_processes([1], False)
