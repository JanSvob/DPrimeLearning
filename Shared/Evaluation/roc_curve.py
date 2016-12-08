#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def roc_curve(genuine_scores, impostor_scores, plot_label='roc'):
    # Compute vector lengths
    numGenuine = len(genuine_scores)
    numImpostor = len(impostor_scores)

    fars = []
    frrs = []
    mfrrs = []
    threshs = []

    minScore = np.min(genuine_scores) - 1.0
    maxScore = np.max(genuine_scores) + 1.0
    step = (maxScore - minScore) / 1000.0

    for i in np.arange(minScore, maxScore, step):
        far = np.sum(impostor_scores <= i) / float(numImpostor)
        frr = np.sum(genuine_scores > i) / float(numGenuine)
        fars.append(far)
        frrs.append(frr)
        mfrrs.append(1.0 - frr)
        threshs.append(i)

    fars = np.array(fars)
    frrs = np.array(frrs)
    threshs = np.array(threshs)

    # Compute EER
    idx = np.argmin(np.absolute(fars - frrs))
    frrEER = np.sum(genuine_scores > threshs[idx]) / float(numGenuine)
    farEER = np.sum(impostor_scores < threshs[idx]) / float(numImpostor)
    eer = (frrEER + farEER) / 2.0

    sns.set_style('darkgrid')
    plt.plot(fars, mfrrs, label=plot_label)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Opreation Curve')
    plt.legend(bbox_to_anchor=(1, 0.5))

    return fars, frrs, eer
