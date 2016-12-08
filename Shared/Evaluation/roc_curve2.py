#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def roc_curve(genuine_scores, impostor_scores, plot_label='roc'):
    # Compute vector lengths
    numGenuine = len(genuine_scores)
    numImpostor = len(impostor_scores)

    tprs = []
    fprs = []
    threshs = []

    minScore = np.min(genuine_scores)
    maxScore = np.max(genuine_scores)
    step = (maxScore - minScore) / 1000.0

    for i in np.arange(minScore, maxScore, step):
        tpr = np.sum(genuine_scores <= i) / float(numGenuine)
        fpr = np.sum(impostor_scores <= i) / float(numImpostor)
        tprs.append(tpr)
        fprs.append(fpr)
        threshs.append(i)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    threshs = np.array(threshs)

    # Compute EER
    idx = np.argmin(np.absolute(1 - tprs - fprs))
    eer1 = np.sum(genuine_scores > threshs[idx]) / float(numGenuine)
    eer2 = np.sum(impostor_scores <= threshs[idx]) / float(numImpostor)
    eer = (eer1 + eer2) / 2.0

    sns.set_style('darkgrid')
    plt.plot(1 - fprs, 1 - tprs, label=plot_label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Opreation Curve')
    plt.legend(bbox_to_anchor=(1, 0.5))

    return tprs, fprs, 1 - eer
