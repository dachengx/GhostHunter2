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
from tqdm import tqdm

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


# def CalPed(w) :
#     stream.FastCalculate(w)
#     return stream.ChannelInfo.Ped
# 
# 
# VCalPed = np.vectorize(CalPed, signature="(n)->()")
# Peds = VCalPed(Waveform["Waveform"])


# @numba.jit
# def MakeWaveImage(EventIDs, ChannelIDs, Charges, θ_index, φ_index) :
#     WaveImage = np.zeros((nEvents, 30, 30, 1029), dtype=np.float32)
#     for i in range(len(EventIDs)) :
#         channelid = ChannelIDs[i]
#         WaveImage[EventIDs[i] - 1][φ_index[channelid]][θ_index[channelid]] = Ped[i] - Waveform["Waveform"][i]
#     return WaveImage


class WaveData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    WaveImage = tables.Float32Col(pos=1, shape=(30, 30, 1029))
    HitImage = tables.Int8Col(pos=2, shape=(30, 30, 1029))
    Alpha = tables.BoolCol(pos=3)


# ChargeImage = MakeChargeImage(Waveform["EventID"], Waveform["ChannelID"], Charges, θ_index, φ_index)
pre_file = tables.open_file(opt, mode="w", filters=tables.Filters(complevel=9))
TrainTable = pre_file.create_table("/", "TrainTable", WaveData, "TrainTable")
Row = TrainTable.row

EventIDs, EventID_index = np.unique(Waveform["EventID"], return_index=True)
EventID_index = np.append(EventID_index, len(Waveform))
EventIDs_pe, EventID_peindex = np.unique(PETruth["EventID"], return_index=True)
EventID_peindex = np.append(EventID_peindex, len(PETruth))

stream = JPwaptool(1029, 150, 200, 10, 30)
# for i in tqdm(range(nEvents)) :
for i in tqdm(range(30000)) :
    Row["EventID"] = EventIDs[i]
    Row["Alpha"] = ParticleTruth["Alpha"][i]
    WaveImage = np.zeros((30,30,1029), dtype=np.float32)
    for j in range(EventID_index[i], EventID_index[i+1]) :
        channelid = Waveform["ChannelID"][j]
        stream.FastCalculate(Waveform["Waveform"][i])
        WaveImage[φ_index[channelid]][θ_index[channelid]] = stream.ChannelInfo.Ped - Waveform["Waveform"][i]
    HitImage = np.zeros((30,30,1029), dtype=np.int8)
    for k in range(EventID_peindex[i], EventID_peindex[i+1]) :
        channelid = PETruth["ChannelID"][k]
        # print("{0}: {1}".format(channelid, PETruth["PETime"][k]))
        HitImage[φ_index[channelid]][θ_index[channelid]][PETruth["PETime"][k]] += 1
    Row["WaveImage"] = WaveImage
    Row["HitImage"] = HitImage
    Row.append()
TrainTable.flush()
pre_file.close()
