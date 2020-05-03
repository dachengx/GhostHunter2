import numpy as np


def calAUC(answer, truth):
    Answer_Truth = list(zip(answer, truth))
    rank = np.array([values2 for values1, values2 in sorted(Answer_Truth, key=lambda x:x[0])])
    rankIndex = np.array([i + 1 for i in range(len(rank)) if rank[i] == 1])
    posNum = rank.sum()
    negNum = len(rank) - posNum
    auc = (rankIndex.sum() - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    return auc
