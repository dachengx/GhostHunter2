# -*- coding: utf-8 -*-

import numpy as np
import tables
import torch
import matplotlib.pyplot as plt
from DataIO import ReadParticleType
from sklearn.metrics import roc_curve, auc
from CNN_Module import Net_1
from torchviz import make_dot

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 16
plt.rcParams['axes.unicode_minus'] = False

def calAUC(answer, truth):
    Answer_Truth = list(zip(answer, truth))
    rank = np.array([values2 for values1, values2 in sorted(Answer_Truth, key=lambda x:x[0])])
    rankIndex = np.array([i + 1 for i in range(len(rank)) if rank[i] == 1])
    posNum = rank.sum()
    negNum = len(rank) - posNum
    auc = (rankIndex.sum() - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc

def main(fipt):
    net = Net_1()
    x = torch.rand(1, 1029, 30)
    y = net(x)
    g = make_dot(y)
    g.render('net', view=False)

    ParticleType = ReadParticleType(fipt[0])
    iptfile = tables.open_file(fipt[1], 'r')
    Answer = iptfile.root.Answer.col('Alpha')
    iptfile.close()
    fpr, tpr, thresholds = roc_curve(ParticleType, Answer)
    roc_auc = auc(fpr,tpr)

    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:0.4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='upper left')
    plt.savefig('ROC.png')
    plt.close()
    return

if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('ipt', type=str, nargs='+', help='input file')
    args = psr.parse_args()
    main(args.ipt)
