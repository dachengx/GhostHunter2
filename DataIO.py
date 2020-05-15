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


def ReadTrainSet(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleType = iptfile.root.ParticleType[:]
    TimeProfile = iptfile.root.TimeProfile[:]
    iptfile.close()
    return TimeProfile, ParticleType

def ReadProblemSet(filename):
    iptfile = tables.open_file(filename, 'r')
    EventID = iptfile.root.EventID[:]
    TimeProfile = iptfile.root.TimeProfile[:]
    iptfile.close()
    return EventID, TimeProfile

def GetProfile(PETruth, WindowSize, nChannels):
    Time = np.zeros((nChannels, WindowSize[1] - WindowSize[0]), dtype=np.uint8)
    PETruth = PETruth.query('PETime >= @WindowSize[0] and PETime < @WindowSize[1]')
    PETruth['PETime'] -= WindowSize[0]
    channels_number, channels_indices = np.unique(PETruth['ChannelID'], return_index=True)
    channels_indices = np.append(channels_indices, len(PETruth))
    for j, cid in enumerate(channels_number) :
        this_channel_petruth = PETruth['PETime'][channels_indices[j]:channels_indices[j + 1]]
        petime_number, petime_counts = np.unique(this_channel_petruth, return_counts=True)
        Time[cid][petime_number] = petime_counts
    return Time.T


def MakeTimeProfile(PETruth, WindowSize) :
    nChannels = 30
    PETruth['ChannelID'] -= PETruth['ChannelID'].min()
    tqdm.pandas()
    TimeProfile = vPETruth.groupby('EventID').progress_apply(lambda x:GetProfile(x, WindowSize, nChannels))
    TimeProfile = np.array(TimeProfile.values.tolist())
    return TimeProfile
