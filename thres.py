# -*- coding: utf-8 -*-

import numpy as np
import tables
from tqdm import tqdm
import pandas as pd
import DataIO
from DataIO import ReadPETruth, ReadParticleType, ReadPEGuess
from time import time
from multiprocessing import Pool
from Grader import calAUC


def CalulateStd(PETruth) :
    e_ans, i_ans = np.unique(PETruth['EventID'], return_index=True)
    i_ans = np.append(i_ans, len(PETruth))
    STD = np.empty(len(e_ans), dtype=np.float32)
    for i in tqdm(range(len(e_ans))) :
        this_event_petruth = PETruth[i_ans[i]:i_ans[i + 1]]
        this_event_answer = np.std(this_event_petruth['PETime'])
        STD[i] = this_event_answer
    return {'STD': STD, 'EventID': e_ans}


def ProcessTrainFile(filename) :
    PETruth = ReadPETruth(filename)['Data']
    ParticleType = ReadParticleType(filename)
    return {'STD': CalulateStd(PETruth)['STD'], 'ParticleType': ParticleType}


def classifier(STD, para1, para2) :
    answer = (STD - para1) / para2
    answer = np.vstack([answer , np.ones(answer.shape)]).min(axis=0)
    answer = np.vstack([answer , np.zeros(answer.shape)]).max(axis=0)
    return answer


def train() :
    from IPython import embed
    filenames = ['dataset/pre-{}.h5'.format(i) for i in range(10)]
    with Pool(len(filenames)) as pool :
        Result = pool.map(ProcessTrainFile, filenames)
    STD = np.concatenate([result['STD'] for result in Result])
    ParticleType = np.concatenate([result['ParticleType'] for result in Result])

    para1_series = np.linspace(21, 21.04, 11)
    para2_series = np.linspace(9.1, 9.14, 11)
    grids = np.meshgrid(para1_series, para2_series)
    para1_series = grids[0].flatten()
    para2_series = grids[1].flatten()
    Paras = list(zip(para1_series, para2_series))

    def Scan(para1, para2) :
        return calAUC(classifier(STD, para1, para2), ParticleType)

    with Pool(min(len(para1_series), 250)) as pool :
        Results = pool.starmap(Scan, Paras)

    Results = pd.DataFrame({'para1': para1_series, 'para2': para2_series, 'loss': Results})
    Results = Results.sort_values(by='loss', ascending=True)
    embed()


def main(fopt, fipt, method):
    PETruth = ReadPEGuess(fipt)['Data']
    AnswerFile = tables.open_file(fopt, mode='w', title='AlphaBeta', filters=tables.Filters(complevel=4))
    AnswerTable = AnswerFile.create_table('/', 'Answer', DataIO.AnswerData, 'Answer')
    STD = CalulateStd(PETruth)
    Answer = classifier(STD['STD'], 21.013, 9.102)
    AnswerTable.append(list(zip(STD['EventID'], Answer)))
    AnswerTable.flush()
    AnswerFile.close()


if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt', type=str, help='output file')
    psr.add_argument('ipt', type=str, help='input file')
    psr.add_argument('--met', type=str, help='fitting method')
    args = psr.parse_args()

    main(args.opt, args.ipt, args.met)
