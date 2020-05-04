# -*- coding: utf-8 -*-

import numpy as np
import tables
import pandas as pd
from tqdm import tqdm
import numba

def ReadPETruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    PETruth = iptfile.root.PETruth[:]
    keys = iptfile.root.PETruth.colnames
    PETruth_DataFrame = pd.DataFrame({key: PETruth[key] for key in keys})
    iptfile.close()
    return {'Data': PETruth, 'DataFrame': PETruth_DataFrame}


def ReadParticleTruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleTruth = iptfile.root.ParticleTruth[:]
    keys = iptfile.root.ParticleTruth.colnames
    ParticleTruth_DataFrame = pd.DataFrame({key: ParticleTruth[key] for key in keys})
    iptfile.close()
    return {'Data': ParticleTruth, 'DataFrame': ParticleTruth_DataFrame}


def ReadParticleType(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleType = iptfile.root.ParticleTruth.col('Alpha')
    iptfile.close()
    return ParticleType


def GetProfile(PETruth, WindowLength, nChannels):
    Time = np.zeros((nChannels, WindowLength), dtype=np.uint8)
    channels_number, channels_indices = np.unique(PETruth['ChannelID'], return_index=True)
    channels_indices = np.append(channels_indices, len(PETruth))
    for j, cid in enumerate(channels_number) :
        this_channel_petruth = PETruth['PETime'][channels_indices[j]:channels_indices[j + 1]]
        petime_number, petime_counts = np.unique(this_channel_petruth, return_counts=True)
        Time[cid][petime_number] = petime_counts
    return Time.T


#@numba.jit
def MakeTimeProfile(PETruth, WindowSize) :
    vPETruth = PETruth.query('PETime >=@WindowSize[0] and PETime < @WindowSize[1]').reset_index(drop=True)
    vPETruth['EventID'] -= vPETruth['EventID'].min(); vPETruth['ChannelID'] -= vPETruth['ChannelID'].min()
    vPETruth['PETime'] -= WindowSize[0]; nChannels = len(PETruth['ChannelID'].unique())
    tqdm.pandas(); WindowLength = WindowSize[1]-WindowSize[0]
    TimeProfile = vPETruth.groupby('EventID').progress_apply(lambda x:GetProfile(x, WindowLength, nChannels))
    TimeProfile = np.array(TimeProfile.values.tolist())
    return TimeProfile
