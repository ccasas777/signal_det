import numpy as np
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text, parse_txt


def min_max_norm(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def get_data(data_path):
    text_lines = load_text(data_path)
    time, ch1, ch2 = parse_txt(text_lines)
    time = time - time.min()
    ch1 = min_max_norm(ch1)
    ch2 = min_max_norm(ch2)
    # NOTE: prune the data to first triangle wave in hard code
    sr = 14062500
    shift = 25000
    end = int((1 * 0.02) * sr - shift)
    time, ch1, ch2 = time[end:], ch1[end:], ch2[end:]
    half_cyc_idx = sr * 0.02
    sliced_N = int(time.shape[0] / (half_cyc_idx) + 0.5)
    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    train_n = int((train_ratio * sliced_N) + 0.5)
    val_n = int((val_ratio * sliced_N) + 0.5)
    test_n = int((test_ratio * sliced_N) + 0.5)
    print("File is splited to {}:{}:{}".format(train_n, val_n, test_n))

    train_data = ch1[:sliced_N * train_n]
    val_data = ch1[sliced_N * train_n:(sliced_N * train_n) + (sliced_N * val_n)]
    test_data = ch1[(sliced_N * train_n) + (sliced_N * val_n):]

    return train_data, val_data, test_data


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--file_name', default=None, type=str)
    parser.add_argument('--train_ratio', default=0.7, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    if args.file_name is None:
        file_names = [x for x in os.listdir(args.data_root) if 'txt' in x]
    else:
        file_names = [args.file_name]

    if len(file_names) == 0:
        print('No annotations, exit')
        sys.exit(0)
    for file_name in file_names:
        print('Process %s' % file_name)
        data_path = os.path.join(args.data_root, file_name)
        get_data(data_path)