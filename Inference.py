# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file', type=str)
psr.add_argument('-M', dest='mod', help='netwok model', type=str)
psr.add_argument('-o', dest='opt', help='output dir', type=str)
psr.add_argument('-N', '--frag', dest='frag', type=int)
args = psr.parse_args()
L = args.frag

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import tables
from tqdm import tqdm
import DataIO

device = torch.device(0)

def inference(Wave, Model):
    Ans = np.zeros(len(Wave)).astype(np.float32)
    ProblemData = Data.TensorDataset(torch.from_numpy(Wave).float().cuda(device=device))
    DataLoader = Data.DataLoader(dataset=ProblemData, batch_size=1, shuffle=False, pin_memory=False)
    net = torch.load(Model, map_location=device)
    for i, data in enumerate(tqdm(DataLoader, 0)):
        inputs = data[0]
        inputs = Variable(inputs)
        outputs = net(inputs).cpu().detach().numpy()
        Ans[i] = outputs[0][1]
    return Ans

def main(filename, Model, SavePath):
    torch.cuda.init()
    torch.cuda.empty_cache()

    EventID, Wave = DataIO.ReadProblemSet(filename)
    Answer = np.zeros(len(EventID)).astype(np.float32)

    N = len(Wave)
    for A in range(L+1):
        torch.cuda.empty_cache()
        a = N//L*A; b = min(N//L*(A+1), N)
        Ans = inference(Wave[a:b], Model)
        Answer[a:b] = Ans

    AnswerFile = tables.open_file(SavePath, mode='w', filters=tables.Filters(complevel=4))
    AnswerTable = AnswerFile.create_table('/', 'Answer', DataIO.AnswerData)
    AnswerTable.append(list(zip(EventID, Answer)))
    AnswerTable.flush()
    AnswerFile.close()
    return

if __name__ == '__main__':
    main(args.ipt, args.mod, args.opt)
