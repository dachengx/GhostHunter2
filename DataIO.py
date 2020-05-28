# -*- coding: utf-8 -*-

import numpy as np
import tables
import pandas as pd
from tqdm import tqdm
import numba


class AnswerData(tables.IsDescription) :
    EventID = tables.Int64Col(pos=0)
    Alpha = tables.Float32Col(pos=1)


def ReadPETruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    PETruth = iptfile.root.PETruth[:]
    keys = iptfile.root.PETruth.colnames
    PETruth_DataFrame = pd.DataFrame({key: PETruth[key] for key in keys})
    iptfile.close()
    return {'Data': PETruth, 'DataFrame': PETruth_DataFrame}


def ReadPEGuess(filename) :
    iptfile = tables.open_file(filename, 'r')
    PEGuess = iptfile.root.PEGuess[:]
    keys = iptfile.root.PEGuess.colnames
    PEGuess_DataFrame = pd.DataFrame({key: PEGuess[key] for key in keys})
    iptfile.close()
    return {'Data': PEGuess, 'DataFrame': PEGuess_DataFrame}


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


def ReadWaveform(filename) :
    iptfile = tables.open_file(filename, 'r')
    Waveform = iptfile.root.Waveform[:]
    keys = iptfile.root.Waveform.colnames
    Waveform_DataFrame = pd.DataFrame({key: list(Waveform[key]) for key in keys})
    iptfile.close()
    return {'Data': Waveform, 'DataFrame': Waveform_DataFrame}


#def ReadTrainSet(filename) :
#    iptfile = tables.open_file(filename, 'r')
#    ParticleType = iptfile.root.ParticleType[:]
#    TimeProfile = iptfile.root.TimeProfile[:]
#    iptfile.close()
#    return TimeProfile, ParticleType


#def ReadProblemSet(filename):
#    iptfile = tables.open_file(filename, 'r')
#    EventID = iptfile.root.EventID[:]
#    TimeProfile = iptfile.root.TimeProfile[:]
#    iptfile.close()
#    return EventID, TimeProfile


def ReadTrainSet(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleType = iptfile.root.ParticleType[:]
    Wave = iptfile.root.Wave[:]
    iptfile.close()
    return Wave, ParticleType


def ReadProblemSet(filename):
    iptfile = tables.open_file(filename, 'r')
    EventID = iptfile.root.EventID[:]
    Wave = iptfile.root.Wave[:]
    iptfile.close()
    return EventID, Wave


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

def GetWaveform(Waveform, nChannels):
    Wave = np.zeros((nChannels, len(Waveform.loc[0]['Waveform'])), dtype=np.int16)
    Wave[Waveform['ChannelID']] = np.array(Waveform['Waveform'].values.tolist())
    return Wave.T

def MakeWaveform(Waveform) :
    nChannels = 30
    Waveform['ChannelID'] -= Waveform['ChannelID'].min()
    tqdm.pandas()
    Wave = Waveform.groupby('EventID').progress_apply(lambda x:GetWaveform(x, nChannels))
    Wave = np.array(Wave.values.tolist())
    return Wave


def MakeTimeProfile(PETruth, WindowSize) :
    nChannels = 30
    PETruth['ChannelID'] -= PETruth['ChannelID'].min()
    tqdm.pandas()
    TimeProfile = PETruth.groupby('EventID').progress_apply(lambda x:GetProfile(x, WindowSize, nChannels))
    TimeProfile = np.array(TimeProfile.values.tolist())
    return TimeProfile
