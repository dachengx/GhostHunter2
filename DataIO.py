import numpy as np
import tables
import pandas as pd
import numba


def ReadPETruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    PETruth = iptfile.root.PETruth[:]
    keys = iptfile.root.PETruth.colnames
    PETruth_DataFrame = pd.DataFrame({key: PETruth[key] for key in keys})
    iptfile.close()
    return {"Data": PETruth, "DataFrame": PETruth_DataFrame}


def ReadParticleTruth(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleTruth = iptfile.root.ParticleTruth[:]
    keys = iptfile.root.ParticleTruth.colnames
    ParticleTruth_DataFrame = pd.DataFrame({key: ParticleTruth[key] for key in keys})
    iptfile.close()
    return {"Data": ParticleTruth, "DataFrame": ParticleTruth_DataFrame}


def ReadParticleType(filename) :
    iptfile = tables.open_file(filename, 'r')
    ParticleType = iptfile.root.ParticleTruth.col("Alpha")
    iptfile.close()
    return ParticleType


@numba.jit
def MakeTimeProfile(WindowSize, nEvents, nChannels, PETruth) :
    TimeProfile = np.zeros((nEvents, nChannels, WindowSize), dtype=np.uint8)
    for i in range(len(PETruth)) :
        TimeProfile[PETruth["EventID"][i] - 1][PETruth["ChannelID"][i]][PETruth["PETime"][i]] += 1
    return TimeProfile
