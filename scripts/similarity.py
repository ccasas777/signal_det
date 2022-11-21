import numpy as np
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import sys
from pathlib import Path
from scipy.signal import butter, lfilter

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text, load_json, parse_txt
"""
command line:
    python3 scripts/similarity.py --config=./config/example.json --low_pass True  --is_draw True
"""


def conv1d(src, pattern):
    p_n, s_n = pattern.shape[0], src.shape[0]
    x = np.zeros_like(src)
    windowed_pattern = pattern
    print('Process conv1D with a known pattern ...')
    for i in range(s_n):
        windowed_src = src[i:p_n + i]
        if np.size(windowed_src) != np.size(windowed_pattern):
            windowed_pattern = windowed_pattern[:np.size(windowed_src)]
            x[i:p_n + i] += windowed_src * windowed_pattern
            break
        else:
            x[i:p_n + i] += windowed_src * windowed_pattern
    return x


def lowpass(x):

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    # create a butter worth filter
    order = 5
    fs = 1250.0
    cutoff = 30
    y = butter_lowpass_filter(x, cutoff, fs, order)
    y -= np.mean(y, axis=-1)
    return y


def min_max_norm(x):
    return (x - x.min(keepdims=True, axis=-1)) / (
        x.max(keepdims=True, axis=-1) - x.min(keepdims=True, axis=-1))


def run_similarity(time, src, patterns, root_dir, window=None, is_draw=False):
    # if is_conv:
    #     src = conv1d(src, patterns)

    s_n = src.shape[0]
    src = src.reshape([-1, s_n])
    stride = 1
    sim_thres, interval_per = 10., 0.5
    kernel_size = patterns.shape[1]

    normed_src = min_max_norm(src)
    filted_src = lowpass(normed_src)

    steps = (s_n - kernel_size + 1) / stride
    steps += 1  # remainder
    tmp, poss_intervals = [], []
    windowed_pattern = min_max_norm(patterns)

    eps = 1e-9
    for i in tqdm(range(int(steps))):
        windowed_src = filted_src[:, i:kernel_size + i]
        # last step
        if i == steps - 1:
            windowed_pattern = windowed_pattern[:, :np.size(windowed_src)]
        windowed_src = min_max_norm(windowed_src)
        curr_dist = np.sqrt(
            np.sum(np.square(windowed_pattern - windowed_src), axis=-1) + eps)
        tmp.append(curr_dist)
        if i != 0 and len(tmp) % kernel_size == 0:
            tmp = np.asarray(tmp).T
            # hard threshold
            mask = tmp < sim_thres
            valid_sim = np.float32(mask)
            guessed_interval = np.sum(valid_sim, axis=-1) / kernel_size
            votes = np.sum((guessed_interval > 0.5).astype(np.float32) /
                           np.size(guessed_interval))
            if votes > interval_per:
                poss_intervals.append(i)
            tmp = []
    convert2time = time[poss_intervals]
    print('Finish evaluating')
    print('Totoal find %i' % len(poss_intervals))
    print('Window size %i' % kernel_size)
    np.save(os.path.join(root_dir, "index.npy"), poss_intervals)
    if is_draw:
        # OR draw src
        min_max = [np.min(filted_src), np.max(filted_src)]
        for t in convert2time:
            plt.plot([t, t], min_max, 'r-')
        plt.plot(time, filted_src[0])
        plt.plot(time, filted_src[0])
        plt.grid()
        plt.savefig('foo.png')


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--low_pass', default=True, type=bool)
    parser.add_argument('--is_draw', default=True, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print('Run signal detection with a known pattern')
    print(f"Use following config to produce results: {args.config}.")
    cfg = load_json(args.config)

    if os.path.isfile(cfg["pattern"]):
        patterns = np.stack([np.load(cfg["pattern"])])
    else:
        patterns = np.stack([
            np.load(os.path.join(cfg["pattern"], p))
            for p in os.listdir(cfg["pattern"])
        ])
    time, raw_data, triangle_waves = parse_txt(load_text(cfg["raw_data"]))
    root_dir = os.path.split(cfg["raw_data"])[0]
    run_similarity(time, raw_data, patterns, root_dir, args.low_pass,
                   args.is_draw)
