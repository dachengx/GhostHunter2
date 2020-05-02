# -*- coding: utf-8 -*-

import sys
import re
import numpy as np
import h5py
import itertools as it
import argparse

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--met', type=str, help='fitting method')
args = psr.parse_args()

def main(fopt, fipt, method):
    opdt = np.dtype([('EventID', np.uint32), ('Alpha', np.float32)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        petru = ipt['PETruth']
        e_ans, i_ans = np.unique(petru['EventID'], return_index=True)
        e_num = e_ans.shape[0]
        dt = np.zeros(e_num, dtype=opdt)
        dt['EventID'] = e_ans
        for eid, j, i0, i in zip(e_ans, range(e_num), np.nditer(i_ans), it.chain(np.nditer(i_ans[1:]), [len(petru)])):
            pet = petru[i0:i]['PETime']
            dt[j]['Alpha'] = min(max((np.std(pet)-24)/8, 0), 1)
            print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*j)//e_num)*'-', (19-(20*j)//e_num)*' ', 100 * ((j+1) / e_num)), end='' if j != e_num-1 else '\n')
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.met)
