# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file prefix', type=str)
psr.add_argument('-o', '--outputdir', dest='opt', help='output_dir')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=64)
psr.add_argument('-P', '--pretrained', dest='pretained_model', nargs='?', type=str, const='')
args = psr.parse_args()
SavePath = args.opt
filename = args.ipt
BATCHSIZE = args.BAT
Model = args.pretained_model

import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import DataIO
from CNN_Module import Net_1

import os
import time

if not os.path.exists(SavePath):
    os.makedirs(SavePath)
localtime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
training_record_name = SavePath + 'training_record_' + localtime
testing_record_name = SavePath + 'testing_record_' + localtime
training_record = open((training_record_name + '.txt'), 'a+')
testing_record = open((testing_record_name + '.txt'), 'a+')

torch.cuda.init()
torch.cuda.empty_cache()
device = torch.device(0)

loss_set = torch.nn.CrossEntropyLoss()

TimeProfile, ParticleType = DataIO.ReadTrainSet(filename)
TimeProfile_train, TimeProfile_test, ParticleType_train, ParticleType_test = train_test_split(TimeProfile, ParticleType, test_size=0.05, random_state=42)
train_data = Data.TensorDataset(torch.from_numpy(TimeProfile_train).float().cuda(device=device), 
                                torch.from_numpy(ParticleType_train).long().cuda(device=device))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
test_data = Data.TensorDataset(torch.from_numpy(TimeProfile_test).float().cuda(device=device),
                                torch.from_numpy(ParticleType_test).long().cuda(device=device))
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
trial_data = Data.TensorDataset(torch.from_numpy(TimeProfile_test[0:1000]).float().cuda(device=device),
                                torch.from_numpy(ParticleType_test[0:1000]).long().cuda(device=device))
trial_loader = Data.DataLoader(dataset=trial_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

def testing(test_loader) :
    batch_count = 0
    loss_sum = 0
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        loss = loss_set(outputs, labels)
        loss_sum += loss
        batch_count += 1
    return loss_sum / batch_count

if os.path.exists(Model) :
    net = torch.load(Model, map_location=device)
    lr = 5e-4
else :
    net = Net_1().cuda(device)
    lr = 1e-4
optimizer = optim.Adam(net.parameters(), lr=lr)
checking_period = np.int(0.25 * (len(TimeProfile_train) / BATCHSIZE))

training_result = []
testing_result = []
for epoch in range(25):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_set(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        if (i + 1) % checking_period == 0:    # print every 2000 mini-batches
            print('[%d, %5d] running_loss: %.3f' %(epoch + 1, i + 1, running_loss / checking_period))
            training_record.write('%.3f ' % ((running_loss / checking_period)))
            training_result.append((running_loss / checking_period))
            running_loss = 0.0

    # checking results in testing_s
    if epoch % 4 == 0:
        test_performance = testing(test_loader)
        print('epoch ', str(epoch), ' test:', test_performance)
        testing_record.write('%4f ' % (test_performance))
        testing_result.append(test_performance)
        # saving network
        save_name = SavePath + '_epoch' + str(epoch) + '_loss' + '%.4f' % (test_performance)
        torch.save(net, save_name)

print(training_result)
print(testing_result)

training_record.close()
testing_record.close()
