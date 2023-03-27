from tqdm import tqdm
from evaluate_multiple import train_evaluate
import os
import json
import tensorflow as tf

def fix_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(tf_session)


def main():
    fix_gpu()
    # with open("tests_process2.json") as f:
    with open("tests.json") as f:
        tests = json.load(f)
    os.makedirs("new_results/", exist_ok=True)
    for test in tqdm(tests):
        print(test)
        key = tuple(test.items())
        if not os.path.exists(f'new_results/{key}.pkl'):
            train_evaluate(test)
        else:
            print(f"Skipping {key}")


if __name__ == "__main__":
    main()
