# -*- coding: utf-8 -*-

import numpy as np
import tables
from tqdm import tqdm
from DataIO import ReadPETruth, ReadParticleType
from time import time
from multiprocessing import Pool
from IPython import embed
from Grader import calAUC


class AnswerData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    Alpha = tables.Float32Col(pos=1)


def CalulateStd(PETruth) :
    e_ans, i_ans = np.unique(PETruth['EventID'], return_index=True)
    i_ans = np.append(i_ans, len(PETruth))
    STD = np.empty(len(e_ans), dtype=np.float32)
    for i in tqdm(range(len(e_ans))) :
        this_event_petruth = PETruth[i_ans[i]:i_ans[i + 1]]
        this_event_answer = np.std(this_event_petruth["PETime"])
        STD[i] = this_event_answer
    return {"STD": STD, "EventID": e_ans}


def ProcessTrainFile(filename) :
    PETruth = ReadPETruth(filename)["Data"]
    ParticleType = ReadParticleType(filename)
    return {"STD": CalulateStd(PETruth)["STD"], "ParticleType": ParticleType}


def train() :
    filenames = ["dataset/pre-{}.h5".format(i) for i in range(10)]
    with Pool(len(filenames)) as pool :
        Result = pool.map(ProcessTrainFile, filenames)
    STD = np.concatenate([result["STD"] for result in Result])
    ParticleType = np.concatenate([result["ParticleType"] for result in Result])
    return np.polyfit(STD, ParticleType, deg=3)


def main(fopt, fipt, para):
    PETruth = ReadPETruth(fipt)["Data"]
    AnswerFile = tables.open_file(fopt, mode="w", title="AlphaBeta", filters=tables.Filters(complevel=4))
    AnswerTable = AnswerFile.create_table("/", "Answer", AnswerData, "Answer")
    classifier = np.poly1d(para)
    STD = CalulateStd(PETruth)
    Answer = classifier(STD["STD"])
    AnswerTable.append(list(zip(STD["EventID"], Answer)))
    AnswerTable.flush()
    AnswerFile.close()


if __name__ == '__main__':
    import argparse
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt', type=str, help='output file')
    psr.add_argument('ipt', type=str, help='input file')
    psr.add_argument('--met', type=str, help='fitting method')
    args = psr.parse_args()
    para = train()
    main(args.opt, args.ipt, para)
