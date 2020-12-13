import pandas as pd
import math
import numpy as np
from meta import ROOT_PATH
from time import gmtime, strftime
import os


def get_mean_std(df: pd.DataFrame, feat_name, norm_name):
    mean = []
    std = []
    m = df.mean()
    s = df.std()
    for i, n in enumerate(feat_name):
        if n in norm_name:
            mean.append(m.get(n, 0.))
            tmp = s.get(n, 1)
            if math.isnan(tmp):
                tmp = 1
            std.append(tmp)
        else:
            mean.append(0)
            std.append(1)

    return np.asarray(mean), np.asarray(std)


def get_feat_ind(source: list, target: list):
    return [source.index(i) for i in target]


def make_dir(task_name):
    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', task_name)
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return summary_path, save_path


if __name__ == '__main__':
    from util.data.feat import *

    print(get_feat_ind(FEAT_LIST, USER_FEAT))
