# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=64)
psr.add_argument('-P', '--pretrained', dest='pretained_model', type=str, default="")
args = psr.parse_args()
BATCHSIZE = args.BAT
Model = args.pretained_model

import os
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import DataIO
from CNN_Module import Net_1

def loss():
    pass


def main(fopt, fipt):
    PETruth = DataIO.ReadPETruth(fipt)["DataFrame"]
    ParticleType = DataIO.ReadParticleType(filename)
    WindowSize = 1000; nEvents = len(PETruth['EventID'].unique()); nChannels = len(PETruth['ChannelID'].unique())
    TimeProfile = DataIO.MakeTimeProfile(WindowSize, nEvents, nChannels, PETruth)
    TimeProfile_train, TimeProfile_test, ParticleType_train, ParticleType_test = train_test_split(TimeProfile, ParticleType, test_size=0.05, random_state=42)
    train_data = Data.TensorDataset(torch.from_numpy(TimeProfile_train).cuda(device=device).float(), 
                                    torch.from_numpy(ParticleType_train).cuda(device=device).float())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
    test_data = Data.TensorDataset(torch.from_numpy(TimeProfile_test).cuda(device=device).float(), 
                                    torch.from_numpy(ParticleType_test).cuda(device=device).float())
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
    trial_data = Data.TensorDataset(torch.from_numpy(TimeProfile_test[0:1000]).cuda(device=device).float(),
                                    torch.from_numpy(ParticleType_test[0:1000]).cuda(device=device).float())
    trial_loader = Data.DataLoader(dataset=trial_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)
    if os.path.exists(Model) :
        net = torch.load(Model, map_location=device)
        loss = testing(trial_loader)
        lr = 5e-4
    else :
        loss = 10000
        while(loss > 100) :
            net = Net_1().to(device)
            loss = testing(trial_loader)
            print("Trying initial parameters with loss={:.2f}".format(loss))
        lr = 1e-2


if __name__ == '__main__':
    main(args.opt, args.ipt)