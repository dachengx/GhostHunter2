#!/usr/bin/env python3
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('opt', help='output')
# psr.add_argument('-N', '--NWaves', dest='Nwav', type=int, help='entries of events')
psr.add_argument('-D', '--Device', dest='dev', type=str, default='cpu')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=64)
args = psr.parse_args()
ipt = args.ipt
opt = args.opt
BATCHSIZE = args.BAT

import tables
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
device = torch.device(args.dev)

from S2CNN_Net import S2ConvNet_original
from IPython import embed


def load_data(batch_size):
    with tables.open_file(ipt) as fip :
        dataset = fip.root.TrainTable[:]
    train_data, test_data, train_labels, test_labels = train_test_split(dataset["ChargeImage"], dataset["Alpha"], test_size=0.05, random_state=42)
    train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1], train_data.shape[2]))
    test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1], test_data.shape[2]))
    train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels).long())
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = data_utils.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels).long())
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_dataset, test_dataset


PMTPosition = np.loadtxt("PMT_Position.txt", skiprows=8)
PMTPosition = pd.DataFrame({"ChannelID": PMTPosition[:, 0].astype(np.int16), "X": PMTPosition[:, 1], "Y": PMTPosition[:, 2], "Z": PMTPosition[:, 3]}).sort_values(by='ChannelID', ignore_index=True)
PMTPosition["R"] = np.sqrt(PMTPosition["X"]**2 + PMTPosition["Y"]**2 + PMTPosition["Z"]**2)
PMTPosition["ρ"] = np.sqrt(PMTPosition["X"]**2 + PMTPosition["Y"]**2)
PMTPosition["θ"] = np.arccos(PMTPosition["Z"] / PMTPosition["R"])
PMTPosition["φ"] = np.pi * 2 * (PMTPosition["Y"] < 0) + np.sign(PMTPosition["Y"] + 1e-5) * np.arccos(PMTPosition["X"] / PMTPosition["ρ"])
PMT_grid = tuple(zip(PMTPosition["θ"], PMTPosition["φ"]))

train_loader, test_loader, train_dataset, _ = load_data(BATCHSIZE)

classifier = S2ConvNet_original(PMT_grid)
classifier.to(device)
print("#params", sum(x.numel() for x in classifier.parameters()))

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-3)

NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        classifier.train()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            epoch + 1, NUM_EPOCHS, i + 1, len(train_dataset) // BATCHSIZE,
            loss.item()), end="")
    print("")
    correct = 0
    total = 0
    for images, labels in test_loader:

        classifier.eval()

        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()

    print('Test Accuracy: {0}'.format(100 * correct / total))
embed()
