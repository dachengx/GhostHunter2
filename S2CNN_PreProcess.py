#!/usr/bin/env python3
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('opt', help='output')
psr.add_argument('-N', '--NWaves', dest='Nwav', type=int, help='entries of waves')
args = psr.parse_args()
ipt = args.ipt
opt = args.opt

import DataIO
import tables
import numpy as np
from JPwaptool import JPwaptool
import numba
import pandas as pd

from IPython import embed

PMTPosition = np.loadtxt("PMT_Position.txt", skiprows=8)
PMTPosition = pd.DataFrame({"ChannelID": PMTPosition[:, 0].astype(np.int16), "X": PMTPosition[:, 1], "Y": PMTPosition[:, 2], "Z": PMTPosition[:, 3]}).sort_values(by='ChannelID', ignore_index=True)
PMTPosition["R"] = np.sqrt(PMTPosition["X"]**2 + PMTPosition["Y"]**2 + PMTPosition["Z"]**2)
PMTPosition["ρ"] = np.sqrt(PMTPosition["X"]**2 + PMTPosition["Y"]**2)
PMTPosition["θ"] = np.arccos(PMTPosition["Z"] / PMTPosition["R"])
PMTPosition["φ"] = np.pi * 2 * (PMTPosition["Y"] < 0) + np.sign(PMTPosition["Y"] + 1e-5) * np.arccos(PMTPosition["X"] / PMTPosition["ρ"])
θ_index = np.round(PMTPosition["θ"] / np.pi * 30).astype(np.int)
PMTPosition["θ_index"] = θ_index
φ_index = np.round(PMTPosition["φ"] / (2 * np.pi) * 30).astype(np.int)
PMTPosition["φ_index"] = φ_index

ParticleTruth = DataIO.ReadParticleTruth("/srv/abpid/dataset/pre-0.h5")
ParticleTruth, ParticleTruth_DataFrame = ParticleTruth["Data"], ParticleTruth["DataFrame"]
PETruth = DataIO.ReadPETruth("/srv/abpid/dataset/pre-0.h5")
PETruth, PETruth_DataFrame = PETruth["Data"], PETruth["DataFrame"]
Waveform = DataIO.ReadWaveform("/srv/abpid/dataset/pre-0.h5")
Waveform, Waveform_DataFrame = Waveform["Data"], Waveform["DataFrame"]
nEvents = Waveform["EventID"][-1]

stream = JPwaptool(1029, 150, 600, 10, 30)


def CalCharge(w) :
    stream.FastCalculate(w)
    return stream.ChannelInfo.Charge


VCalCharge = np.vectorize(CalCharge, signature="(n)->()")
Charges = VCalCharge(Waveform["Waveform"])


@numba.jit
def MakeChargeImage(EventIDs, ChannelIDs, Charges, θ_index, φ_index) :
    ChargeImage = np.zeros((nEvents, 30, 30), dtype=np.float32)
    for i in range(len(EventIDs)) :
        channelid = ChannelIDs[i]
        ChargeImage[EventIDs[i] - 1][φ_index[channelid]][θ_index[channelid]] = Charges[i]
    return ChargeImage


class ChargeData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    ChargeImage = tables.Float32Col(pos=1, shape=(30, 30))
    Alpha = tables.BoolCol(pos=2)


ChargeImage = MakeChargeImage(Waveform["EventID"], Waveform["ChannelID"], Charges, θ_index, φ_index)
pre_file = tables.open_file(opt, mode="w", filters=tables.Filters(complevel=5))
TrainTable = pre_file.create_table("/", "TrainTable", ChargeData, "TrainTable")
Row = TrainTable.row
for i in range(nEvents) :
    Row["EventID"] = ParticleTruth["EventID"][i]
    Row["Alpha"] = ParticleTruth["Alpha"][i]
    Row["ChargeImage"] = Charges[i]
    Row.append()
TrainTable.flush()
pre_file.close()
