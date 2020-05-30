# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file', type=str)
psr.add_argument('-M', dest='mod', help='netwok model', type=str)
psr.add_argument('-o', dest='opt', help='output dir', type=str)
args = psr.parse_args()

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import tables
from tqdm import tqdm
import DataIO

device = torch.device(1)

def main(filename, Model, SavePath):
    torch.cuda.init()
    torch.cuda.empty_cache()

    EventID, Wave = DataIO.ReadProblemSet(filename)
    Answer = np.zeros(len(EventID)).astype(np.float32)

    Answer = np.zeros(len(Wave)).astype(np.float32)
    ProblemData = Data.TensorDataset(torch.from_numpy(Wave).float())
    DataLoader = Data.DataLoader(dataset=ProblemData, batch_size=1, shuffle=False, pin_memory=False)
    net = torch.load(Model, map_location=device)
    for i, data in enumerate(tqdm(DataLoader, 0)):
        torch.cuda.empty_cache()
        inputs = data[0]
        inputs = Variable(inputs.cuda(device=device))
        outputs = net(inputs).cpu().detach().numpy()
        Answer[i] = outputs[0][1]

    AnswerFile = tables.open_file(SavePath, mode='w', filters=tables.Filters(complevel=4))
    AnswerTable = AnswerFile.create_table('/', 'Answer', DataIO.AnswerData)
    AnswerTable.append(list(zip(EventID, Answer)))
    AnswerTable.flush()
    AnswerFile.close()
    return

if __name__ == '__main__':
    main(args.ipt, args.mod, args.opt)
