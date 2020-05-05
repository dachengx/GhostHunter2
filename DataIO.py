# -*- coding: utf-8 -*-

import numpy as np
import tables
import pandas as pd
from pandarallel import pandarallel
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


def GetTime(jPETruth, TimeProfile):
    this_channel_petruth = jPETruth['PETime']
    petime_number, petime_counts = np.unique(this_channel_petruth, return_counts=True)
    TimeProfile[jPETruth['EventID'].iat[0]][jPETruth['ChannelID'].iat[0]][petime_number] = petime_counts


def GetProfile(iPETruth, TimeProfile):
    iPETruth.groupby('ChannelID').apply(lambda x:GetTime(x, TimeProfile))


def MakeTimeProfile(PETruth, WindowSize) :
    vPETruth = PETruth.query('PETime >=@WindowSize[0] and PETime < @WindowSize[1]').reset_index(drop=True)
    vPETruth['EventID'] -= vPETruth['EventID'].min(); vPETruth['ChannelID'] -= vPETruth['ChannelID'].min()
    vPETruth['PETime'] -= WindowSize[0]; WindowLength = WindowSize[1]-WindowSize[0]
    TimeProfile = np.zeros((vPETruth['EventID'].max()+1, vPETruth['ChannelID'].max()+1, WindowLength), dtype=np.uint8)
    pandarallel.initialize(nb_workers=4, progress_bar=True)
    vPETruth.groupby('EventID').parallel_apply(lambda x:GetProfile(x, TimeProfile))
    return TimeProfile.transpose((0, 2, 1))
