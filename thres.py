# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
args = psr.parse_args()

import numpy as np
import tables
from tqdm import tqdm
from DataIO import ReadPETruth


class AnswerData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    Alpha = tables.Float32Col(pos=1)


def StdMethod(PETruth, AnswerRow) :
    e_ans, i_ans = np.unique(PETruth['EventID'], return_index=True)
    i_ans = np.append(i_ans, len(PETruth) - 1)
    for i in tqdm(range(len(e_ans))) :
        this_event_petruth = PETruth[i_ans[i]:i_ans[i + 1]]
        this_event_answer = min(max((np.std(this_event_petruth['PETime']) - 24) / 8, 0), 1)
        AnswerRow['EventID'] = e_ans[i]
        AnswerRow['Alpha'] = this_event_answer
        AnswerRow.append()


def main(fopt, fipt, method):
    PETruth = ReadPETruth(fipt)["Data"]
    AnswerFile = tables.open_file(fopt, mode="w", title="AlphaBeta", filters=tables.Filters(complevel=4))
    AnswerTable = AnswerFile.create_table("/", "Answer", AnswerData, "Answer")
    StdMethod(PETruth, AnswerTable.row)
    AnswerTable.flush()
    AnswerFile.close()


if __name__ == '__main__':
    main(args.opt, args.ipt, args.met)
