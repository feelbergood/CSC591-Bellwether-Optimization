from __future__ import division, print_function

import os
import random
import sys
import warnings
from glob import glob

import numpy as np
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from pathlib import Path

with warnings.catch_warnings():
    # Shut those god damn warnings up!
    warnings.filterwarnings("ignore")

root = os.getcwd()
dir = os.path.abspath(os.path.join(root, "datasets"))
if root not in sys.path:
    sys.path.append(root)


class _Data:

    def __init__(self, dataName='ant', type='jur'):
        if type == 'jur':
            dir = os.path.join(root, "data/Jureczko")
        elif type == 'nasa':
            dir = os.path.join(root, "data/mccabe")
        elif type == 'aeeem':
            dir = os.path.join(root, "data/AEEEM")
        elif type == "relink":
            dir = os.path.join(root, "data/Relink")
        elif type == 'other':
            dir = os.path.join(root, "data/other/")

        self.data = glob(os.path.join(dir, dataName, "*.csv"))


class NASA:
    "NASA"

    def __init__(self):
        self.projects = {}
        for file in ["cm", "jm", "kc", "mc", "mw"]:
            self.projects.update({file: _Data(dataName=file, type='nasa')})


class Jureczko:
    "Apache"

    def __init__(self):
        self.projects = {}
        for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                     'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
            self.projects.update({file: _Data(dataName=file, type='jur')})


class AEEEM:
    "AEEEM"

    def __init__(self):
        self.projects = {}
        for file in ["EQ", "JDT", "LC", "ML", "PDE"]:
            self.projects.update({file: _Data(dataName=file, type='aeeem')})


class ReLink:
    "RELINK"

    def __init__(self):
        self.projects = {}
        for file in ["Apache", "Safe", "Zxing"]:
            self.projects.update({file: _Data(dataName=file, type='relink')})


def get_all_projects():
    all = dict()
    for community in [Jureczko, AEEEM, ReLink, NASA]:
        all.update({community.__doc__: community().projects})
    return all


def abcd(actual, predicted, distribution, as_percent=True):
    actual = [1 if a > 0 else 0 for a in actual]

    def stringify(lst):
        try:
            return [str(int(a)) for a in lst]
        except ValueError:
            return [str(a) for a in lst]

    try:
        fpr, tpr, thresholds = roc_curve(actual, distribution)
        auroc = round(roc_auc_score(actual, distribution), 2)
        cutoff = np.random.uniform(0.27, 0.31)
        for a, b, c in zip(fpr, tpr, thresholds):
            if a < cutoff:
                threshold = c
        predicted = [1 if val > threshold else 0 for val in distribution]
    except:
        auroc = 0
        predicted = [1 if val > 0 else 0 for val in predicted]

    c_mtx = confusion_matrix(actual, predicted)

    "Probablity of Detection: Pd"
    try:
        p_d = c_mtx[1][1] / (c_mtx[1][1] + c_mtx[1][0])  # TP/(TP+FN)
    except:
        p_d = 0

    "Probability of False Alarm: Pf"
    try:
        p_f = c_mtx[0][1] / (c_mtx[0][1] + c_mtx[0][0])  # FP/(FP+TN)
    except:
        p_f = 0

    "Precision"
    try:
        p_r = c_mtx[1][1] / (c_mtx[1][1] + c_mtx[0][1])  # TP/(TP+FP)
        if not np.isfinite(p_r): p_r = 0
    except:
        p_r = 0

    "Recall (Same as Pd)"
    r_c = p_d

    "F1 measure"
    try:
        f1 = 2 * c_mtx[1][1] / (2 * c_mtx[1][1] + c_mtx[0][1] + 1 * c_mtx[1][0])  # F1 = 2*TP/(2*TP+FP+FN)
    except:
        f1 = 0

    "G-Score"
    e_d = 2 * p_d * (1 - p_f) / (1 + p_d - p_f)
    g = np.sqrt(p_d - p_d * p_f)  # Harmonic Mean between True positive rate and True negative rate

    try:
        auroc = round(roc_auc_score(actual, distribution), 2)
    except ValueError:
        auroc = 0

    ed = np.sqrt(0.7 * (1 - p_d) ** 2 + 0.3 * p_f ** 2)
    # e_d = 1 / ((0.5 / p_d) + (0.5 / (1 - p_f)))
    e_d = 2 * p_d * (1 - p_f) / (1 + p_d - p_f)
    g = np.sqrt(p_d - p_d * p_f)  # Harmonic Mean between True positive rate and True negative rate
    # set_trace()
    if np.isnan(p_d or p_f or p_r or r_c or f1 or e_d or g or auroc):
        return 0, 0, 0, 0, 0, 0, 0, 0
    if as_percent is True:
        return p_d * 100, p_f * 100, p_r * 100, r_c * 100, f1 * 100, e_d * 100, g * 100, auroc * 100
    else:
        return p_d, p_f, p_r, r_c, f1, e_d, g, auroc


def rf_model(source, target, seed):
    clf = RandomForestClassifier(n_estimators=seed, random_state=1)
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:, 1]


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (test_instance[tgt[:-1]].std())
    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) + [tgt[-1]]
    return new_train[columns], new_test[columns]


def list2dataframe(lst):
    data = [pandas.read_csv(elem) for elem in lst]
    return pandas.concat(data, ignore_index=True)


def predict_defects(train, test, seed):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test, seed)
    return actual, predicted, distr


def bellw(source, target, fname, verbose=True, n_rep=30):
    result = dict()
    return result

if __name__ == "__main__":
    print("test")